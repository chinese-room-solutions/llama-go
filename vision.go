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

// visionConfig holds configuration for VisionContext creation.
type visionConfig struct {
	useGPU    bool
	threads   int
	flashAttn string
}

// VisionOption configures VisionContext creation.
type VisionOption func(*visionConfig)

// WithVisionGPU enables or disables GPU offloading for the vision encoder.
// Default: true.
func WithVisionGPU(useGPU bool) VisionOption {
	return func(c *visionConfig) {
		c.useGPU = useGPU
	}
}

// WithVisionThreads sets the number of CPU threads for vision encoding.
// Default: uses the model's thread count.
func WithVisionThreads(threads int) VisionOption {
	return func(c *visionConfig) {
		c.threads = threads
	}
}

// WithVisionFlashAttn sets flash attention mode for vision encoding.
// Values: "auto" (default), "enabled", "disabled".
func WithVisionFlashAttn(flashAttn string) VisionOption {
	return func(c *visionConfig) {
		c.flashAttn = flashAttn
	}
}

// VisionContext wraps an mtmd_context for multimodal (vision) inference.
// It is created from a Model with a path to an mmproj GGUF file.
//
// VisionContext is NOT thread-safe — each goroutine/worker should have
// its own VisionContext instance. Multiple VisionContexts can share the
// same underlying Model.
//
// Example:
//
//	vision, err := model.NewVisionContext("/path/to/mmproj.gguf")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer vision.Close()
type VisionContext struct {
	ptr    unsafe.Pointer // mtmd_context*
	model  *Model
	mu     sync.Mutex
	closed bool
}

// NewVisionContext creates a multimodal context for vision inference.
// mmprojPath must point to a valid mmproj GGUF file matching the model.
func (m *Model) NewVisionContext(mmprojPath string, opts ...VisionOption) (*VisionContext, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.modelPtr == nil {
		return nil, fmt.Errorf("model has been closed")
	}

	cfg := visionConfig{
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
		return nil, fmt.Errorf("failed to create vision context: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	v := &VisionContext{
		ptr:   ptr,
		model: m,
	}
	runtime.SetFinalizer(v, (*VisionContext).Close)

	return v, nil
}

// Close releases the vision context resources.
func (v *VisionContext) Close() error {
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
