package ggmlutil

import "fmt"

func DisabledTypeName(t int) string { return fmt.Sprintf("ggml-disabled-%d", t) }

func RawBytes(n, blockSize, typeSize int) int {
	if blockSize <= 0 || typeSize <= 0 {
		return 0
	}
	return (n / blockSize) * typeSize
}
