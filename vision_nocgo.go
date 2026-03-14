//go:build !cgo

package llama

// VisionContext is a stub for non-CGO builds.
// Actual implementation requires CGO (see vision.go).
type VisionContext struct{}
