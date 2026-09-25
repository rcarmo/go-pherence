//go:build !linux || (!amd64 && !arm64)

package nvidia

// Other ABIs retain the typed purego binding.
func bindFixedCUDAKernelLauncher(lib uintptr) {}
