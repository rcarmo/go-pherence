package ggmlutil

import (
	"fmt"
	"github.com/rcarmo/go-pherence/internal/checked"
)

func DisabledTypeName(t int) string { return fmt.Sprintf("ggml-disabled-%d", t) }

// RawBytes rejects incomplete blocks and overflowing extents before a caller
// allocates a native encoded tensor. Zero denotes invalid/empty dimensions.
func RawBytes(n, blockSize, typeSize int) int {
	if n <= 0 || blockSize <= 0 || typeSize <= 0 || n%blockSize != 0 {
		return 0
	}
	bytes, ok := checked.MulInt(n/blockSize, typeSize)
	if !ok {
		return 0
	}
	return bytes
}
