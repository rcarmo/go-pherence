package gpu

import "github.com/rcarmo/go-pherence/tensor"

// Conv1D dispatches 1D convolution, using GPU if available or CPU fallback.
// input: [inChannels * inLength] channel-first
// weight: [outChannels * inChannels * kernelSize]
// bias: [outChannels] (may be nil)
// output: [outChannels * outLength]
func Conv1D(output, input, weight, bias []float32, inChannels, inLength, outChannels, kernelSize, stride, padding int) {
	outLength := (inLength+2*padding-kernelSize)/stride + 1
	if outLength <= 0 || len(input) < inChannels*inLength || len(weight) < outChannels*inChannels*kernelSize || len(output) < outChannels*outLength {
		return
	}

	// TODO: GPU fast path when Conv1D PTX kernel is available
	// For now: CPU fallback only
	tensor.Conv1DFlat(output, input, weight, bias, inChannels, inLength, outChannels, kernelSize, stride, padding)
}
