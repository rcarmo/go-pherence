//go:build !spacemitort || !cgo || !linux

package spacemitort

import "fmt"

// Session is a placeholder when the vendor SpacemiT ONNX Runtime C API is not
// available at build time. Build with -tags spacemitort on a configured K-series
// system to enable the cgo-backed implementation.
type Session struct{}

func NewSession(modelPath string, opts Options) (*Session, error) {
	return nil, fmt.Errorf("spacemit ORT session support not built; rebuild with -tags spacemitort on a system with ONNX Runtime and SpacemiT EP headers/libraries")
}

func (s *Session) Close() {}

func (s *Session) Run1(inputName string, input []float32, shape []int64, outputName string, outputElems int) ([]float32, error) {
	return nil, fmt.Errorf("spacemit ORT session support not built")
}
