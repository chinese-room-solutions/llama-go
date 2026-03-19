# llama-go Build System
#
# Usage: make <target> [VAR=val]
#
# On Windows (Git Bash / MSYS2): delegates to build_win.ps1
# On Linux / macOS: runs native build commands
#
# Variables:
#   BUILD_TYPE         Backend type: cublas, hipblas, metal, openblas, clblas, blis (default: none)
#   CUDA_ARCHITECTURES CUDA compute capabilities (e.g. 86, "86;89")
#   CMAKE_ARGS         Extra CMake arguments (appended to auto-detected ones)
#   GPU_TESTS          Set to "true" to run GPU tests

# -- Platform detection -------------------------------------------------------

ifndef UNAME_S
UNAME_S := $(shell uname -s)
endif

ifndef UNAME_P
UNAME_P := $(shell uname -p)
endif

ifndef UNAME_M
UNAME_M := $(shell uname -m)
endif

IS_WIN := $(findstring MINGW,$(UNAME_S))$(findstring MSYS,$(UNAME_S))$(findstring CYGWIN,$(UNAME_S))

# -- Shared library extension per platform ------------------------------------

ifeq ($(UNAME_S),Darwin)
  SHLIB_EXT := dylib
else
  SHLIB_EXT := so
endif

# -- Paths --------------------------------------------------------------------

INCLUDE_PATH := $(abspath ./)
LIBRARY_PATH := $(abspath ./)

# -- Windows: delegate to PowerShell -----------------------------------------

ifdef IS_WIN

PS := $(shell command -v pwsh 2>/dev/null || command -v powershell 2>/dev/null)

.PHONY: build clean help

build:
	$(PS) -NoProfile -ExecutionPolicy Bypass -File build_win.ps1
	$(PS) -NoProfile -ExecutionPolicy Bypass -File build_wrapper.ps1

clean:
	$(PS) -NoProfile -Command "Remove-Item -ErrorAction SilentlyContinue *.obj, *.lib, *.dll, *.exp"
	$(PS) -NoProfile -Command "if (Test-Path build) { Remove-Item -Recurse -Force build }"

help:
	@echo ""
	@echo "  llama-go Build System (Windows)"
	@echo "  ================================"
	@echo ""
	@echo "  Targets:"
	@echo "    build    Build all (cmake + wrapper + copy libs)"
	@echo "    clean    Remove build artifacts"
	@echo ""

# -- Linux / macOS: native build ----------------------------------------------

else

CCV  := $(shell $(CC) --version | head -n 1)
CXXV := $(shell $(CXX) --version | head -n 1)

# -- Mac ARM detection --------------------------------------------------------

ifeq ($(UNAME_S),Darwin)
	ifneq ($(UNAME_P),arm)
		SYSCTL_M := $(shell sysctl -n hw.optional.arm64 2>/dev/null)
		ifeq ($(SYSCTL_M),1)
			warn := $(warning Your arch is reported as x86_64 but appears to be ARM64. This may cause suboptimal performance.)
		endif
	endif
endif

# -- Compile flags ------------------------------------------------------------

BUILD_TYPE ?=

CFLAGS   = -I./llama.cpp -I. -O3 -DNDEBUG -std=c11 -fPIC
CXXFLAGS = -I./llama.cpp -I. -I./llama.cpp/common -I./common \
           -I./llama.cpp/ggml/include -I./llama.cpp/include \
           -I./llama.cpp/vendor -I./llama.cpp/tools/mtmd \
           -O3 -DNDEBUG -std=c++17 -fPIC
LDFLAGS  =

# Warnings
CFLAGS   += -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion \
            -Wshadow -Wstrict-prototypes -Wpointer-arith -Wno-unused-function
CXXFLAGS += -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function

# -- OS-specific flags --------------------------------------------------------

ifneq (,$(filter Linux Darwin FreeBSD NetBSD OpenBSD Haiku,$(UNAME_S)))
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif

ifndef LLAMA_NO_ACCELERATE
ifeq ($(UNAME_S),Darwin)
	CFLAGS  += -DGGML_USE_ACCELERATE
	LDFLAGS += -framework Accelerate
endif
endif

# -- Architecture-specific flags ----------------------------------------------

ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686))
	CFLAGS += -march=native -mtune=native
endif

ifneq ($(filter ppc64%,$(UNAME_M)),)
	POWER9_M := $(shell grep "POWER9" /proc/cpuinfo)
	ifneq (,$(findstring POWER9,$(POWER9_M)))
		CFLAGS   += -mcpu=power9
		CXXFLAGS += -mcpu=power9
	endif
	ifeq ($(UNAME_M),ppc64)
		CXXFLAGS += -std=c++23 -DGGML_BIG_ENDIAN
	endif
endif

ifneq ($(filter aarch64%,$(UNAME_M)),)
	CFLAGS   += -mcpu=native
	CXXFLAGS += -mcpu=native
endif

ifneq ($(filter armv6%,$(UNAME_M)),)
	CFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access
endif

ifneq ($(filter armv7%,$(UNAME_M)),)
	CFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
endif

ifneq ($(filter armv8%,$(UNAME_M)),)
	CFLAGS += -mfp16-format=ieee -mno-unaligned-access
endif

# -- Backend-specific flags ---------------------------------------------------

EXTRA_TARGETS =
GGML_CUDA_OBJ_PATH = ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/ggml-cuda.cu.o

ifeq ($(BUILD_TYPE),openblas)
	CMAKE_ARGS += -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DBLAS_INCLUDE_DIRS=/usr/include/openblas
endif

ifeq ($(BUILD_TYPE),blis)
	CMAKE_ARGS += -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=FLAME
endif

ifeq ($(BUILD_TYPE),cublas)
	CMAKE_ARGS += -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_GRAPHS=ON
	CXXFLAGS   += -DGGML_USE_CUDA
	ifdef CUDA_ARCHITECTURES
		CMAKE_ARGS += -DCMAKE_CUDA_ARCHITECTURES="$(CUDA_ARCHITECTURES)"
	endif
	EXTRA_TARGETS += llama.cpp/ggml-cuda.o
endif

ifeq ($(BUILD_TYPE),hipblas)
	ROCM_HOME ?= /opt/rocm
	CXX = $(ROCM_HOME)/llvm/bin/clang++
	CC  = $(ROCM_HOME)/llvm/bin/clang
	GPU_TARGETS    ?= gfx900,gfx90a,gfx1030,gfx1031,gfx1100
	AMDGPU_TARGETS ?= $(GPU_TARGETS)
	CMAKE_ARGS += -DGGML_HIP=ON -DAMDGPU_TARGETS="$(AMDGPU_TARGETS)" -DGPU_TARGETS="$(GPU_TARGETS)"
	CXXFLAGS   += -DGGML_USE_HIP
	EXTRA_TARGETS  += llama.cpp/ggml-cuda.o
	GGML_CUDA_OBJ_PATH = ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/ggml-cuda.cu.o
endif

ifeq ($(BUILD_TYPE),clblas)
	CMAKE_ARGS    += -DGGML_OPENCL=ON
	EXTRA_TARGETS += llama.cpp/ggml-opencl.o
endif

ifeq ($(BUILD_TYPE),metal)
	CGO_LDFLAGS   += "-framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders"
	CMAKE_ARGS    += -DGGML_METAL=ON
	EXTRA_TARGETS += llama.cpp/ggml-metal.o
endif

ifdef CLBLAST_DIR
	CMAKE_ARGS += -DCLBlast_dir=$(CLBLAST_DIR)
endif

ifdef LLAMA_OPENBLAS
	CFLAGS  += -DGGML_USE_OPENBLAS -I/usr/local/include/openblas
	LDFLAGS += -lopenblas
endif

ifdef LLAMA_GPROF
	CFLAGS   += -pg
	CXXFLAGS += -pg
endif

# -- Test config --------------------------------------------------------------

ifeq ($(GPU_TESTS),true)
	CGO_LDFLAGS = "-lcublas -lcudart -L/usr/local/cuda/lib64/"
	TEST_LABEL  = gpu
else
	TEST_LABEL  = !gpu
endif

# -- Print build info ---------------------------------------------------------

$(info )
$(info llama-go build info)
$(info   UNAME_S:      $(UNAME_S))
$(info   UNAME_P:      $(UNAME_P))
$(info   UNAME_M:      $(UNAME_M))
$(info   BUILD_TYPE:   $(BUILD_TYPE))
$(info   CMAKE_ARGS:   $(CMAKE_ARGS))
$(info   EXTRA_TARGETS:$(EXTRA_TARGETS))
$(info   CC:           $(CCV))
$(info   CXX:          $(CXXV))
$(info )

# -- Intermediate object targets ----------------------------------------------

.PHONY: build test clean help libbinding.a

llama.cpp/ggml.o:
	@echo "==> Configuring and building llama.cpp core..."
	mkdir -p build
	cd build && CC="$(CC)" CXX="$(CXX)" cmake ../llama.cpp -DBUILD_SHARED_LIBS=OFF $(CMAKE_ARGS) -DLLAMA_CURL=OFF \
		&& VERBOSE=1 cmake --build . --config Release --target ggml llama \
		&& cp -rf ggml/src/CMakeFiles/ggml-base.dir/ggml.c.o ../llama.cpp/ggml.o

llama.cpp/ggml-cuda.o: llama.cpp/ggml.o
	cd build && cp -rf "$(GGML_CUDA_OBJ_PATH)" ../llama.cpp/ggml-cuda.o

llama.cpp/ggml-opencl.o: llama.cpp/ggml.o
	cd build && cp -rf CMakeFiles/ggml.dir/ggml-opencl.cpp.o ../llama.cpp/ggml-opencl.o

llama.cpp/ggml-metal.o: llama.cpp/ggml.o
	cd build && cp -rf CMakeFiles/ggml.dir/ggml-metal.m.o ../llama.cpp/ggml-metal.o

wrapper.o:
	@echo "==> Compiling wrapper..."
	$(CXX) $(CXXFLAGS) wrapper.cpp -o wrapper.o -c $(LDFLAGS)

# -- Main build target --------------------------------------------------------

libbinding.a: llama.cpp/ggml.o wrapper.o $(EXTRA_TARGETS)
	@echo "==> Building cmake targets (common, mtmd)..."
	cd build && cmake --build . --target common
	cd build && cmake --build . --target mtmd
	@echo "==> Creating libbinding.a..."
	ar crs libbinding.a wrapper.o $(EXTRA_TARGETS)
	@echo "==> Copying libraries..."
	cp build/common/libcommon.a .
ifneq (,$(findstring -DBUILD_SHARED_LIBS=ON,$(CMAKE_ARGS)))
	@echo "Copying shared libraries..."
	cp build/bin/libmtmd.$(SHLIB_EXT) .
	cp build/bin/libllama.$(SHLIB_EXT) .
	cp build/bin/libggml.$(SHLIB_EXT) .
	cp build/bin/libggml-base.$(SHLIB_EXT) .
	cp build/bin/libggml-cpu.$(SHLIB_EXT) .
ifeq ($(BUILD_TYPE),cublas)
	cp build/bin/libggml-cuda.$(SHLIB_EXT) .
endif
else
	@echo "Copying static libraries..."
	cp build/tools/mtmd/libmtmd.a .
	cp build/src/libllama.a .
	cp build/ggml/src/libggml.a .
	cp build/ggml/src/libggml-base.a .
	cp build/ggml/src/libggml-cpu.a .
ifeq ($(BUILD_TYPE),cublas)
	cp build/ggml/src/ggml-cuda/libggml-cuda.a . 2>/dev/null || \
		cp build/ggml/src/libggml-cuda.a . 2>/dev/null || \
		echo "Warning: libggml-cuda.a not found"
endif
endif
	@echo "    Libraries ready"

build: libbinding.a

# -- Test ---------------------------------------------------------------------

ggllm-test-model.bin:
	wget -q https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q2_K.gguf -O ggllm-test-model.bin

test: ggllm-test-model.bin libbinding.a
	C_INCLUDE_PATH=${INCLUDE_PATH} CGO_LDFLAGS=${CGO_LDFLAGS} LIBRARY_PATH=${LIBRARY_PATH} \
		TEST_MODEL=ggllm-test-model.bin \
		go run github.com/onsi/ginkgo/v2/ginkgo --label-filter="$(TEST_LABEL)" --flake-attempts 5 -v -r ./...

# -- Clean --------------------------------------------------------------------

clean:
	rm -rf *.o *.a *.so *.dylib llama.cpp/*.o build

# -- Help ---------------------------------------------------------------------

help:
	@echo ""
	@echo "  llama-go Build System"
	@echo "  ====================="
	@echo ""
	@echo "  Targets:"
	@echo "    build        Build all (cmake + wrapper + libraries)"
	@echo "    test         Run tests (downloads test model if needed)"
	@echo "    clean        Remove all build artifacts"
	@echo ""
	@echo "  Variables:"
	@echo "    BUILD_TYPE         Backend: cublas, hipblas, metal, openblas, clblas, blis"
	@echo "    CUDA_ARCHITECTURES CUDA compute capabilities (e.g. 86)"
	@echo "    CMAKE_ARGS         Extra CMake arguments"
	@echo "    GPU_TESTS          Set to 'true' for GPU tests"
	@echo ""

endif
