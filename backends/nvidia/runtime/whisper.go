package nvidia

import (
	"fmt"
	"unsafe"
)

var (
	fnWhisperMelSpectrogram      CUfunction
	fnWhisperConv1DK3S1          CUfunction
	fnWhisperConv1DK3S2          CUfunction
	fnWhisperAttentionFull       CUfunction
	fnWhisperAttentionFullOnline CUfunction
	fnWhisperCrossAttention      CUfunction
	fnWhisperAttentivePool       CUfunction
	fnWhisperRowAffine           CUfunction
	fnWhisperRowBias             CUfunction
	fnWhisperTranspose           CUfunction
	fnWhisperGELUTanh            CUfunction
)

func validWhisperMatrixBuffers(out, in *Buffer, rows, cols int) bool {
	return whisperF32Extent(out, rows, cols) && whisperF32Extent(in, rows, cols)
}

// WhisperRowAffineBuffer computes out[row,col] = x[row,col]*weight[col]+bias[col].
func WhisperRowAffineBuffer(out, x, weight, bias *Buffer, rows, cols int) error {
	if fnWhisperRowAffine == 0 || !megaModuleOK || !validWhisperMatrixBuffers(out, x, rows, cols) || weight == nil || bias == nil || weight.Size < cols*4 || bias.Size < cols*4 {
		return fmt.Errorf("invalid Whisper row-affine device buffers")
	}
	r, c := uint32(rows), uint32(cols)
	n := rows * cols
	return LaunchKernel(fnWhisperRowAffine, uint32((n+255)/256), 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&x.Ptr), unsafe.Pointer(&weight.Ptr), unsafe.Pointer(&bias.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&r), unsafe.Pointer(&c))
}

// WhisperRowBiasBuffer adds bias[col] to each row in-place.
func WhisperRowBiasBuffer(x, bias *Buffer, rows, cols int) error {
	if fnWhisperRowBias == 0 || !megaModuleOK || !validWhisperMatrixBuffers(x, x, rows, cols) || bias == nil || bias.Size < cols*4 {
		return fmt.Errorf("invalid Whisper row-bias device buffers")
	}
	r, c := uint32(rows), uint32(cols)
	n := rows * cols
	return LaunchKernel(fnWhisperRowBias, uint32((n+255)/256), 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&x.Ptr), unsafe.Pointer(&bias.Ptr), unsafe.Pointer(&r), unsafe.Pointer(&c))
}

// WhisperTransposeBuffer transposes row-major [rows,cols] into [cols,rows].
func WhisperTransposeBuffer(out, in *Buffer, rows, cols int) error {
	if fnWhisperTranspose == 0 || !megaModuleOK || !validWhisperMatrixBuffers(out, in, rows, cols) {
		return fmt.Errorf("invalid Whisper transpose device buffers")
	}
	r, c := uint32(rows), uint32(cols)
	n := rows * cols
	return LaunchKernel(fnWhisperTranspose, uint32((n+255)/256), 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&in.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&r), unsafe.Pointer(&c))
}

// WhisperGELUTanhBuffer applies the tanh GELU approximation in-place.
func WhisperGELUTanhBuffer(x *Buffer, n int) error {
	if fnWhisperGELUTanh == 0 || !megaModuleOK || !whisperF32Extent(x, n) {
		return fmt.Errorf("invalid Whisper GELU device buffer")
	}
	nn := uint32(n)
	return LaunchKernel(fnWhisperGELUTanh, uint32((n+255)/256), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&nn))
}

// WhisperMelSpectrogramBuffer launches the correctness-first fused Whisper mel
// PTX kernel over GPU-resident buffers. Output is mel-major [numMels,numFrames].
func WhisperMelSpectrogramBuffer(out, audio, window, filters *Buffer, numFrames, fftSize, hopLength, numMels, numBins int) error {
	if fnWhisperMelSpectrogram == 0 || !megaModuleOK || !validWhisperMel(out, audio, window, filters, numFrames, fftSize, hopLength, numMels, numBins) {
		return fmt.Errorf("invalid Whisper mel device buffers")
	}
	frames, fs, hop, mels, bins := uint32(numFrames), uint32(fftSize), uint32(hopLength), uint32(numMels), uint32(numBins)
	return LaunchKernel(fnWhisperMelSpectrogram, uint32(numFrames), 1, 1, uint32(numMels), 1, 1, 0,
		unsafe.Pointer(&out.Ptr), unsafe.Pointer(&audio.Ptr), unsafe.Pointer(&window.Ptr), unsafe.Pointer(&filters.Ptr),
		unsafe.Pointer(&frames), unsafe.Pointer(&fs), unsafe.Pointer(&hop), unsafe.Pointer(&mels), unsafe.Pointer(&bins))
}

func WhisperConv1DK3S1Buffer(out, in, weight, bias *Buffer, inChannels, inLength, outChannels, outLength int) error {
	return whisperConv1DBuffer(fnWhisperConv1DK3S1, out, in, weight, bias, inChannels, inLength, outChannels, outLength, "s1")
}

func WhisperConv1DK3S2Buffer(out, in, weight, bias *Buffer, inChannels, inLength, outChannels, outLength int) error {
	return whisperConv1DBuffer(fnWhisperConv1DK3S2, out, in, weight, bias, inChannels, inLength, outChannels, outLength, "s2")
}

func whisperConv1DBuffer(fn CUfunction, out, in, weight, bias *Buffer, inChannels, inLength, outChannels, outLength int, name string) error {
	if fn == 0 || !megaModuleOK || !validWhisperConv(out, in, weight, bias, inChannels, inLength, outChannels, outLength, name) {
		return fmt.Errorf("invalid Whisper conv1d %s device buffers", name)
	}
	ic, il, oc, ol := uint32(inChannels), uint32(inLength), uint32(outChannels), uint32(outLength)
	var biasPtr CUdeviceptr
	if bias != nil {
		biasPtr = bias.Ptr
	}
	gridX := uint32((outLength + 255) / 256)
	return LaunchKernel(fn, gridX, uint32(outChannels), 1, 256, 1, 1, 0,
		unsafe.Pointer(&out.Ptr), unsafe.Pointer(&in.Ptr), unsafe.Pointer(&weight.Ptr), unsafe.Pointer(&biasPtr),
		unsafe.Pointer(&ic), unsafe.Pointer(&il), unsafe.Pointer(&oc), unsafe.Pointer(&ol))
}

func WhisperAttentionFullBuffer(out, q, k, v *Buffer, seqQ, seqKV, numHeads, headDim int, scale float32) error {
	return whisperAttentionBuffer(fnWhisperAttentionFull, out, q, k, v, seqQ, seqKV, numHeads, headDim, scale, "full", 128)
}

// WhisperAttentionFullOnlineBuffer launches the no-score-materialization
// candidate kernel. Keep the default dispatcher on WhisperAttentionFullBuffer
// unless this variant proves faster while staying within the current parity
// tolerance on live benchmarks.
func WhisperAttentionFullOnlineBuffer(out, q, k, v *Buffer, seqQ, seqKV, numHeads, headDim int, scale float32) error {
	return whisperAttentionBuffer(fnWhisperAttentionFullOnline, out, q, k, v, seqQ, seqKV, numHeads, headDim, scale, "full_online", 128)
}

func WhisperCrossAttentionBuffer(out, q, k, v *Buffer, decLen, encLen, numHeads, headDim int, scale float32) error {
	return whisperAttentionBuffer(fnWhisperCrossAttention, out, q, k, v, decLen, encLen, numHeads, headDim, scale, "cross", 1)
}

func whisperAttentionBuffer(fn CUfunction, out, q, k, v *Buffer, seqQ, seqKV, numHeads, headDim int, scale float32, name string, blockX uint32) error {
	if fn == 0 || !megaModuleOK || !validWhisperAttention(out, q, k, v, seqQ, seqKV, numHeads, headDim, scale, name) {
		return fmt.Errorf("invalid Whisper %s attention device buffers", name)
	}
	sq, skv, nh, hd := uint32(seqQ), uint32(seqKV), uint32(numHeads), uint32(headDim)
	return LaunchKernel(fn, uint32(numHeads), uint32(seqQ), 1, blockX, 1, 1, 0,
		unsafe.Pointer(&out.Ptr), unsafe.Pointer(&q.Ptr), unsafe.Pointer(&k.Ptr), unsafe.Pointer(&v.Ptr),
		unsafe.Pointer(&sq), unsafe.Pointer(&skv), unsafe.Pointer(&nh), unsafe.Pointer(&hd), unsafe.Pointer(&scale))
}

func WhisperAttentivePoolBuffer(out, h, attnW, attnB, v *Buffer, channels, length, attnDim int, vBias float32) error {
	if fnWhisperAttentivePool == 0 || !megaModuleOK || out == nil || h == nil || attnW == nil || attnB == nil || v == nil || channels <= 0 || length <= 0 || attnDim <= 0 || !fitsUint32(channels) || !fitsUint32(length) || !fitsUint32(attnDim) {
		return fmt.Errorf("invalid Whisper attentive pool device buffers")
	}
	ch, l, ad := uint32(channels), uint32(length), uint32(attnDim)
	return LaunchKernel(fnWhisperAttentivePool, uint32(channels), 1, 1, 1, 1, 1, 0,
		unsafe.Pointer(&out.Ptr), unsafe.Pointer(&h.Ptr), unsafe.Pointer(&attnW.Ptr), unsafe.Pointer(&attnB.Ptr), unsafe.Pointer(&v.Ptr),
		unsafe.Pointer(&vBias), unsafe.Pointer(&ch), unsafe.Pointer(&l), unsafe.Pointer(&ad))
}
