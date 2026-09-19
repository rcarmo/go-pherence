package tensor

import "github.com/rcarmo/go-pherence/internal/checked"

// Shared preflight keeps dimension arithmetic safe before allocation/indexing.
func conv1DGeometry(inChannels, inLength, outChannels, kernelSize, stride, padding int) (outLength, inputSize, weightSize, outputSize int, ok bool) {
	if inChannels <= 0 || inLength <= 0 || outChannels <= 0 || kernelSize <= 0 || stride <= 0 || padding < 0 {
		return
	}
	twicePad, ok := checked.MulInt(padding, 2)
	if !ok {
		return
	}
	padded, ok := checked.AddInt(inLength, twicePad)
	if !ok || padded < kernelSize {
		return 0, 0, 0, 0, false
	}
	outLength, ok = checked.AddInt((padded-kernelSize)/stride, 1)
	if !ok {
		return
	}
	inputSize, ok = checked.MulInt(inChannels, inLength)
	if !ok {
		return
	}
	rowWeights, ok := checked.MulInt(inChannels, kernelSize)
	if !ok {
		return
	}
	weightSize, ok = checked.MulInt(outChannels, rowWeights)
	if !ok {
		return
	}
	outputSize, ok = checked.MulInt(outChannels, outLength)
	return
}

// Conv1D computes 1D convolution: out[oc][j] = sum_ic sum_k input[ic][j*stride+k-padding] * weight[oc][ic][k] + bias[oc]
//
// input:  [inChannels, inLength]
// weight: [outChannels, inChannels, kernelSize]
// bias:   [outChannels] (may be nil)
// output: [outChannels, outLength]
//
// outLength = (inLength + 2*padding - kernelSize) / stride + 1
func Conv1D(input [][]float32, weight [][][]float32, bias []float32, stride, padding int) [][]float32 {
	if len(input) == 0 || len(weight) == 0 || len(weight[0]) == 0 || len(weight[0][0]) == 0 {
		return nil
	}

	inChannels := len(input)
	inLength := len(input[0])
	outChannels := len(weight)
	kernelSize := len(weight[0][0])

	if stride <= 0 {
		stride = 1
	}
	if padding < 0 {
		padding = 0
	}

	outLength, _, _, _, ok := conv1DGeometry(inChannels, inLength, outChannels, kernelSize, stride, padding)
	if !ok {
		return nil
	}
	for _, row := range input {
		if len(row) != inLength {
			return nil
		}
	}
	for _, outWeights := range weight {
		if len(outWeights) != inChannels {
			return nil
		}
		for _, row := range outWeights {
			if len(row) != kernelSize {
				return nil
			}
		}
	}

	output := make([][]float32, outChannels)
	for oc := range output {
		output[oc] = make([]float32, outLength)
	}

	for oc := 0; oc < outChannels; oc++ {
		for j := 0; j < outLength; j++ {
			var sum float32
			baseIdx := j*stride - padding
			for ic := 0; ic < inChannels && ic < len(weight[oc]); ic++ {
				for k := 0; k < kernelSize; k++ {
					inIdx := baseIdx + k
					if inIdx < 0 || inIdx >= inLength {
						continue // zero-padding
					}
					sum += input[ic][inIdx] * weight[oc][ic][k]
				}
			}
			if bias != nil && oc < len(bias) {
				sum += bias[oc]
			}
			output[oc][j] = sum
		}
	}

	return output
}

// Conv1DFlat computes 1D convolution with flat contiguous buffers for SIMD/GPU friendliness.
//
// input:  [inChannels * inLength] (row-major: channel-first)
// weight: [outChannels * inChannels * kernelSize]
// bias:   [outChannels] (may be nil)
// output: [outChannels * outLength]
func Conv1DFlat(output, input, weight, bias []float32, inChannels, inLength, outChannels, kernelSize, stride, padding int) {
	outLength, inputSize, weightSize, outputSize, ok := conv1DGeometry(inChannels, inLength, outChannels, kernelSize, stride, padding)
	if !ok || len(output) < outputSize || len(input) < inputSize || len(weight) < weightSize {
		return
	}

	for oc := 0; oc < outChannels; oc++ {
		for j := 0; j < outLength; j++ {
			var sum float32
			baseIdx := j*stride - padding
			for ic := 0; ic < inChannels; ic++ {
				wOff := (oc*inChannels + ic) * kernelSize
				iOff := ic * inLength
				for k := 0; k < kernelSize; k++ {
					inIdx := baseIdx + k
					if inIdx < 0 || inIdx >= inLength {
						continue
					}
					sum += input[iOff+inIdx] * weight[wOff+k]
				}
			}
			if bias != nil && oc < len(bias) {
				sum += bias[oc]
			}
			output[oc*outLength+j] = sum
		}
	}
}
