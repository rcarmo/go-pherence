package gpu

import "github.com/rcarmo/go-pherence/tensor"

// Conv1D dispatches 1D convolution, using GPU if available or CPU fallback.
// input: [inChannels * inLength] channel-first
// weight: [outChannels * inChannels * kernelSize]
// bias: [outChannels] (may be nil)
// output: [outChannels * outLength]
func Conv1D(output, input, weight, bias []float32, inChannels, inLength, outChannels, kernelSize, stride, padding int) {
	// Geometry/overflow checks belong to the shared implementation. Computing
	// output length here first used to divide by zero for stride=0.

	// TODO: GPU fast path when Conv1D PTX kernel is available
	// For now: CPU fallback only
	tensor.Conv1DFlat(output, input, weight, bias, inChannels, inLength, outChannels, kernelSize, stride, padding)
}
