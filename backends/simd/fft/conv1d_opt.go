//go:build amd64

package fft

// Conv1DK3S1 performs 1D convolution with kernel_size=3, stride=1, padding=1.
// Optimized for the Whisper conv stem common case.
// input: [inChannels * inLength], weight: [outChannels * inChannels * 3], bias: [outChannels]
// output: [outChannels * inLength]
//
// This pure-Go version uses loop unrolling for the k=3 case.
// The assembly version would use FMA to process 8 output elements per iteration.
func Conv1DK3S1(output, input, weight, bias []float32, inChannels, inLength, outChannels int) {
	if inLength <= 0 || inChannels <= 0 || outChannels <= 0 {
		return
	}
	if len(output) < outChannels*inLength || len(input) < inChannels*inLength || len(weight) < outChannels*inChannels*3 {
		return
	}

	for oc := 0; oc < outChannels; oc++ {
		var biasVal float32
		if bias != nil && oc < len(bias) {
			biasVal = bias[oc]
		}

		outOff := oc * inLength
		for j := 0; j < inLength; j++ {
			sum := biasVal
			for ic := 0; ic < inChannels; ic++ {
				wOff := (oc*inChannels + ic) * 3
				iOff := ic * inLength

				// k=0 (j-1, with zero-pad)
				if j > 0 {
					sum += input[iOff+j-1] * weight[wOff]
				}
				// k=1 (j)
				sum += input[iOff+j] * weight[wOff+1]
				// k=2 (j+1, with zero-pad)
				if j+1 < inLength {
					sum += input[iOff+j+1] * weight[wOff+2]
				}
			}
			output[outOff+j] = sum
		}
	}
}

// Conv1DK3S2 performs 1D convolution with kernel_size=3, stride=2, padding=1.
// output length = (inLength + 1) / 2
func Conv1DK3S2(output, input, weight, bias []float32, inChannels, inLength, outChannels int) {
	outLength := (inLength+2-3)/2 + 1
	if outLength <= 0 || inChannels <= 0 || outChannels <= 0 {
		return
	}
	if len(output) < outChannels*outLength || len(input) < inChannels*inLength || len(weight) < outChannels*inChannels*3 {
		return
	}

	for oc := 0; oc < outChannels; oc++ {
		var biasVal float32
		if bias != nil && oc < len(bias) {
			biasVal = bias[oc]
		}

		outOff := oc * outLength
		for j := 0; j < outLength; j++ {
			sum := biasVal
			base := j*2 - 1 // stride=2, padding=1
			for ic := 0; ic < inChannels; ic++ {
				wOff := (oc*inChannels + ic) * 3
				iOff := ic * inLength

				for k := 0; k < 3; k++ {
					idx := base + k
					if idx >= 0 && idx < inLength {
						sum += input[iOff+idx] * weight[wOff+k]
					}
				}
			}
			output[outOff+j] = sum
		}
	}
}
