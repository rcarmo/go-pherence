//go:build !ggml || !cgo || !linux

package ggmlcompute

import (
	"fmt"

	"github.com/rcarmo/go-pherence/backends/internal/ggmlutil"
)

const (
	Q2K = 10
	Q3K = 11
	Q6K = 14
	Q8K = 15
)

func TypeName(t int) string     { return ggmlutil.DisabledTypeName(t) }
func TypeSize(t int) int        { return 0 }
func BlockSize(t int) int       { return 0 }
func RawBytes(t int, n int) int { return ggmlutil.RawBytes(n, BlockSize(t), TypeSize(t)) }

func QuantizeQ8K(x []float32) ([]byte, error) {
	return nil, fmt.Errorf("ggmlcompute support not built; rebuild with -tags ggml on a system with GGML headers/libraries")
}

func VecDotRowsDirect(qtype int, out []float32, rows []byte, rowBytes int, q8 []byte, n int, nrows int) error {
	return fmt.Errorf("ggmlcompute support not built")
}
