package nvidia

import (
	"fmt"
	"unsafe"
)

var (
	fnWhisperMelSpectrogram CUfunction
	fnWhisperConv1DK3S1     CUfunction
	fnWhisperConv1DK3S2     CUfunction
	fnWhisperAttentionFull  CUfunction
	fnWhisperCrossAttention CUfunction
	fnWhisperAttentivePool  CUfunction
)

// WhisperMelSpectrogramBuffer launches the correctness-first fused Whisper mel
// PTX kernel over GPU-resident buffers. Output is mel-major [numMels,numFrames].
func WhisperMelSpectrogramBuffer(out, audio, window, filters *Buffer, numFrames, fftSize, hopLength, numMels, numBins int) error {
	if fnWhisperMelSpectrogram == 0 || !megaModuleOK || out == nil || audio == nil || window == nil || filters == nil || numFrames <= 0 || fftSize <= 0 || hopLength <= 0 || numMels <= 0 || numBins <= 0 || !fitsUint32(numFrames) || !fitsUint32(fftSize) || !fitsUint32(hopLength) || !fitsUint32(numMels) || !fitsUint32(numBins) {
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
	if fn == 0 || !megaModuleOK || out == nil || in == nil || weight == nil || inChannels <= 0 || inLength <= 0 || outChannels <= 0 || outLength <= 0 || !fitsUint32(inChannels) || !fitsUint32(inLength) || !fitsUint32(outChannels) || !fitsUint32(outLength) {
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
	return whisperAttentionBuffer(fnWhisperAttentionFull, out, q, k, v, seqQ, seqKV, numHeads, headDim, scale, "full")
}

func WhisperCrossAttentionBuffer(out, q, k, v *Buffer, decLen, encLen, numHeads, headDim int, scale float32) error {
	return whisperAttentionBuffer(fnWhisperCrossAttention, out, q, k, v, decLen, encLen, numHeads, headDim, scale, "cross")
}

func whisperAttentionBuffer(fn CUfunction, out, q, k, v *Buffer, seqQ, seqKV, numHeads, headDim int, scale float32, name string) error {
	if fn == 0 || !megaModuleOK || out == nil || q == nil || k == nil || v == nil || seqQ <= 0 || seqKV <= 0 || numHeads <= 0 || headDim <= 0 || !fitsUint32(seqQ) || !fitsUint32(seqKV) || !fitsUint32(numHeads) || !fitsUint32(headDim) {
		return fmt.Errorf("invalid Whisper %s attention device buffers", name)
	}
	sq, skv, nh, hd := uint32(seqQ), uint32(seqKV), uint32(numHeads), uint32(headDim)
	return LaunchKernel(fn, uint32(numHeads), uint32(seqQ), 1, 1, 1, 1, 0,
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
