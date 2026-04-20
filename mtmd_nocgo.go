//go:build !cgo

package llama

// MtmdContext is a stub for non-CGO builds.
// Actual implementation requires CGO (see mtmd.go).
type MtmdContext struct{}
