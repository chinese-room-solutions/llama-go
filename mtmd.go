//go:build cgo

package llama

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

/*
#include "wrapper.h"
#include <stdlib.h>
*/
import "C"

// mtmdConfig holds configuration for MtmdContext creation.
type mtmdConfig struct {
	useGPU    bool
	threads   int
	flashAttn string
}

// MtmdOption configures MtmdContext creation.
type MtmdOption func(*mtmdConfig)

// WithMtmdGPU enables or disables GPU offloading for the multimodal encoder.
// Default: true.
func WithMtmdGPU(useGPU bool) MtmdOption {
	return func(c *mtmdConfig) {
		c.useGPU = useGPU
	}
}

// WithMtmdThreads sets the number of CPU threads for multimodal encoding.
// Default: uses the model's thread count.
func WithMtmdThreads(threads int) MtmdOption {
	return func(c *mtmdConfig) {
		c.threads = threads
	}
}

// WithMtmdFlashAttn sets flash attention mode for multimodal encoding.
// Values: "auto" (default), "enabled", "disabled".
func WithMtmdFlashAttn(flashAttn string) MtmdOption {
	return func(c *mtmdConfig) {
		c.flashAttn = flashAttn
	}
}

// MtmdContext wraps an mtmd_context for multimodal inference (image + audio).
// It is created from a Model with a path to an mmproj GGUF file.
//
// MtmdContext is NOT thread-safe — each goroutine/worker should have
// its own MtmdContext instance. Multiple MtmdContexts can share the
// same underlying Model.
//
// Example:
//
//	mtmd, err := model.NewMtmdContext("/path/to/mmproj.gguf")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer mtmd.Close()
type MtmdContext struct {
	ptr    unsafe.Pointer // mtmd_context*
	model  *Model
	mu     sync.Mutex
	closed bool
}

// NewMtmdContext creates a multimodal context for image + audio inference.
// mmprojPath must point to a valid mmproj GGUF file matching the model.
func (m *Model) NewMtmdContext(mmprojPath string, opts ...MtmdOption) (*MtmdContext, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.modelPtr == nil {
		return nil, fmt.Errorf("model has been closed")
	}

	cfg := mtmdConfig{
		useGPU:    true,
		threads:   runtime.NumCPU(),
		flashAttn: "auto",
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	cPath := C.CString(mmprojPath)
	defer C.free(unsafe.Pointer(cPath))

	var cFlashAttn *C.char
	if cfg.flashAttn != "auto" {
		cFlashAttn = C.CString(cfg.flashAttn)
		defer C.free(unsafe.Pointer(cFlashAttn))
	}

	ptr := C.llama_wrapper_mtmd_init(m.modelPtr, cPath,
		C.bool(cfg.useGPU), C.int(cfg.threads), cFlashAttn)
	if ptr == nil {
		return nil, fmt.Errorf("failed to create mtmd context: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	v := &MtmdContext{
		ptr:   ptr,
		model: m,
	}
	runtime.SetFinalizer(v, (*MtmdContext).Close)

	return v, nil
}

// Close releases the mtmd context resources.
func (v *MtmdContext) Close() error {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.closed {
		return nil
	}
	v.closed = true

	if v.ptr != nil {
		C.llama_wrapper_mtmd_free(v.ptr)
		v.ptr = nil
	}

	runtime.SetFinalizer(v, nil)
	return nil
}
