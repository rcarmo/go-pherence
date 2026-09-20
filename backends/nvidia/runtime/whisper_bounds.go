package nvidia

import (
	"github.com/rcarmo/go-pherence/internal/checked"
	"math"
)

// These kernels linearise float elements in u32 before widening byte offsets.
// Pure host checks deliberately do not inspect or initialize CUDA state.
func whisperF32Extent(b *Buffer, dims ...int) bool {
	if b == nil || b.Ptr == 0 {
		return false
	}
	n := 1
	for _, d := range dims {
		var ok bool
		n, ok = checked.MulInt(n, d)
		if d <= 0 || !ok || !fitsUint32(n) {
			return false
		}
	}
	bytes, ok := checked.MulInt(n, 4)
	return ok && b.Size >= bytes
}
func validWhisperAttention(out, q, k, v *Buffer, sq, skv, heads, dim int, scale float32, name string) bool {
	if sq <= 0 || sq > 65535 || heads <= 0 || heads > 65535 || skv <= 0 || dim <= 0 || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return false
	}
	switch name {
	case "full":
		if dim > 128 || skv > 2048 {
			return false
		}
	case "full_online":
		if dim > 128 {
			return false
		}
	case "cross":
	default:
		return false
	}
	return whisperF32Extent(out, sq, heads, dim) && whisperF32Extent(q, sq, heads, dim) && whisperF32Extent(k, skv, heads, dim) && whisperF32Extent(v, skv, heads, dim)
}
func validWhisperConv(out, in, w, bias *Buffer, ic, il, oc, ol int, name string) bool {
	if oc > 65535 || il <= 0 || il > math.MaxInt32-2 {
		return false
	}
	expected := il
	switch name {
	case "s1":
	case "s2":
		expected = 1 + (il-1)/2
	default:
		return false
	}
	return ol == expected && whisperF32Extent(out, oc, ol) && whisperF32Extent(in, ic, il) && whisperF32Extent(w, oc, ic, 3) && (bias == nil || whisperF32Extent(bias, oc))
}
func validWhisperMel(out, audio, window, filters *Buffer, frames, fft, hop, mels, bins int) bool {
	if frames <= 0 || fft <= 0 || fft > 512 || hop <= 0 || mels <= 0 || mels > 1024 || bins <= 0 || bins > 257 {
		return false
	}
	base, ok := checked.MulInt(frames-1, hop)
	if !ok {
		return false
	}
	samples, ok := checked.AddInt(base, fft)
	return ok && whisperF32Extent(audio, samples) && whisperF32Extent(window, fft) && whisperF32Extent(filters, mels, bins) && whisperF32Extent(out, mels, frames)
}
