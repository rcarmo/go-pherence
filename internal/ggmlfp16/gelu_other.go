//go:build !amd64

package ggmlfp16

var hasF16C = false

func geluFP16MulSIMD(_, _, _ []float32) int { return 0 }
