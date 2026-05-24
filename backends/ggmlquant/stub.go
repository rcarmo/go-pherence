//go:build !ggml || !cgo || !linux

package ggmlquant

import "fmt"

const (
	F32  = 0
	F16  = 1
	Q8_0 = 8
	Q8_K = 15
	Q2_K = 10
	Q3_K = 11
	Q6_K = 14
)

func TypeName(t int) string     { return fmt.Sprintf("ggml-disabled-%d", t) }
func TypeSize(t int) int        { return 0 }
func BlockSize(t int) int       { return 0 }
func VecDotType(t int) int      { return 0 }
func NRows(t int) int           { return 0 }
func HasVecDot(t int) bool      { return false }
func HasFromFloat(t int) bool   { return false }
func RawBytes(t int, n int) int { return 0 }

func QuantizeFromFloat(t int, x []float32) ([]byte, error) {
	return nil, fmt.Errorf("ggmlquant support not built; rebuild with -tags ggml on a system with GGML headers/libraries")
}
func DequantRow(t int, raw []byte, out []float32) error {
	return fmt.Errorf("ggmlquant support not built")
}
func VecDot(t int, xRaw []byte, yRaw []byte, n int) (float32, error) {
	return 0, fmt.Errorf("ggmlquant support not built")
}
func VecDotRows(t int, out []float32, xRows []byte, rowBytes int, yRaw []byte, n int, nrows int) error {
	return fmt.Errorf("ggmlquant support not built")
}
