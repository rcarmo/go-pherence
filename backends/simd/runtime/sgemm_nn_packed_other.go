//go:build !amd64

package simd

// Other architectures retain their existing NN accumulation implementation.
const hasPackedNN = false
