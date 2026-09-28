package qwen3tts

import (
	"fmt"
	"math"
	"runtime"
	"unsafe"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// Candle's CPU convolution reduces each GEMM column in balanced FMA blocks.
// Keep the original portable scalar paths on other architectures until their
// released-model numerical and performance gates can be run natively.
func decoderCandleOrder() bool { return runtime.GOARCH == "amd64" && simd.HasSgemmAsm }

func decoderBlockSize(k int) int {
	const maxK = 512
	if k <= 0 {
		return 0
	}
	blocks := 1 + (k-1)/maxK
	size := k / blocks
	if k%blocks != 0 {
		size++
	}
	return size
}

func decoderProjectionRows(dst, input, weight []float32, rows, cols int) bool {
	if rows <= 0 || cols <= 0 || len(dst) < rows || len(input) != cols || sizeProduct(rows, cols) != len(weight) {
		return false
	}
	if !decoderCandleOrder() {
		return simd.GemvRows(dst, input, weight, rows, cols)
	}
	for r := 0; r < rows; r++ {
		sum := float32(0)
		kc := decoderBlockSize(cols)
		for kk := 0; kk < cols; kk += kc {
			part := float32(0)
			for c := kk; c < min(cols, kk+kc); c++ {
				part = float32(math.FMA(float64(input[c]), float64(weight[r*cols+c]), float64(part)))
			}
			sum += part
		}
		dst[r] = sum
	}
	return true
}

func decoderLinearForward(l talkerLinear, dst, input []float32) error {
	if !decoderCandleOrder() {
		return l.forward(dst, input)
	}
	if l.outDim <= 0 || l.inDim <= 0 || len(dst) != l.outDim || len(input) != l.inDim || sizeProduct(l.outDim, l.inDim) != len(l.weight) || (len(l.bias) != 0 && len(l.bias) != l.outDim) || !decoderProjectionRows(dst, input, l.weight, l.outDim, l.inDim) {
		return fmt.Errorf("invalid Qwen3-TTS decoder linear buffers out/in=%d/%d want %d/%d", len(dst), len(input), l.outDim, l.inDim)
	}
	for i, b := range l.bias {
		dst[i] += b
	}
	return nil
}

func decoderGELUExact(x []float32) {
	if !decoderCandleOrder() {
		simd.GELUExact(x, x)
		return
	}
	for i, v := range x {
		arg := float32(v * float32(math.Sqrt(0.5)))
		e := decoderErff(arg)
		x[i] = float32(float32(e+1)*0.5) * v
	}
}

func (c decoderConv1D) forwardCandleOrder(input []float32, length, groups int) ([]float32, int, error) {
	// The enclosing forward validates input geometry. A group shares its
	// packed kernel only with its own channels, including depthwise groups.
	k := sizeProduct(c.inChannels, c.k)
	if k <= 0 || c.outChannels <= 0 || c.outChannels%groups != 0 || sizeProduct(c.outChannels, k) != len(c.weight) || len(c.bias) != c.outChannels {
		return nil, 0, fmt.Errorf("invalid Qwen3-TTS decoder conv weights")
	}
	const tile = 96
	n := c.outChannels / groups
	a := make([]float32, tile*k)
	result := make([]float32, tile*n)
	outCount := sizeProduct(c.outChannels, length)
	if outCount <= 0 {
		return nil, 0, fmt.Errorf("Qwen3-TTS decoder conv output size overflow")
	}
	out := make([]float32, outCount)
	kc := decoderBlockSize(k)
	pack := make([]float32, kc*16)
	for group := 0; group < groups; group++ {
		for pos := 0; pos < length; pos += tile {
			nr := min(tile, length-pos)
			clear(a)
			clear(result)
			for t := 0; t < nr; t++ {
				for ic := 0; ic < c.inChannels; ic++ {
					for tap := 0; tap < c.k; tap++ {
						src := pos + t - c.dilation*(c.k-1-tap)
						if src >= 0 {
							a[t*k+ic*c.k+tap] = input[(group*c.inChannels+ic)*length+src]
						}
					}
				}
			}
			for kk := 0; kk < k; kk += kc {
				if !simd.SgemmNTGebpWithPack(nr, n, min(kc, k-kk), 1, unsafe.Pointer(&a[kk]), unsafe.Pointer(&c.weight[group*n*k+kk]), unsafe.Pointer(&result[0]), k, k, n, pack) {
					return nil, 0, fmt.Errorf("Qwen3-TTS decoder causal GEMM geometry")
				}
			}
			for t := 0; t < nr; t++ {
				for oc := 0; oc < n; oc++ {
					out[(group*n+oc)*length+pos+t] = result[t*n+oc] + c.bias[group*n+oc]
				}
			}
		}
	}
	return out, length, nil
}

func (c decoderTransConv1D) forwardCandleOrderValidated(input []float32, length int) ([]float32, int, error) {
	if length <= 0 || c.inChannels <= 0 || c.outChannels <= 0 || c.stride <= 0 || c.k < c.stride || sizeProduct(c.inChannels, length) != len(input) || sizeProduct(c.inChannels, c.outChannels, c.k) != len(c.weight) || len(c.bias) != c.outChannels {
		return nil, 0, fmt.Errorf("invalid transposed conv geometry input=%d length=%d channels=%d/%d kernel=%d stride=%d", len(input), length, c.inChannels, c.outChannels, c.k, c.stride)
	}
	outLen := sizeProduct(length, c.stride)
	outCount := sizeProduct(c.outChannels, outLen)
	if outLen < 0 || outCount < 0 {
		return nil, 0, fmt.Errorf("transposed conv output size overflow")
	}
	return c.forwardCandleOrder(input, length, outLen, outCount)
}

func (c decoderTransConv1D) forwardCandleOrder(input []float32, length, outLen, outCount int) ([]float32, int, error) {
	const tile = 96
	width := c.outChannels * c.k
	a := make([]float32, tile*c.inChannels)
	col := make([]float32, tile*width)
	tmp := make([]float32, tile*width)
	out := make([]float32, outCount)
	kc := decoderBlockSize(c.inChannels)
	for pos := 0; pos < length; pos += tile {
		nr := min(tile, length-pos)
		clear(col)
		for t := 0; t < nr; t++ {
			for ic := 0; ic < c.inChannels; ic++ {
				a[t*c.inChannels+ic] = input[ic*length+pos+t]
			}
		}
		for kk := 0; kk < c.inChannels; kk += kc {
			clear(tmp)
			if !simd.SgemmNNTo(tmp, a[kk:], c.weight[kk*width:], nr, width, min(kc, c.inChannels-kk), 1, c.inChannels, width, width) {
				return nil, 0, fmt.Errorf("Qwen3-TTS decoder transposed GEMM geometry")
			}
			for i := range col {
				col[i] += tmp[i]
			}
		}
		for t := 0; t < nr; t++ {
			for oc := 0; oc < c.outChannels; oc++ {
				for tap := 0; tap < c.k && (pos+t)*c.stride+tap < outLen; tap++ {
					out[oc*outLen+(pos+t)*c.stride+tap] += col[(t*c.outChannels+oc)*c.k+tap]
				}
			}
		}
	}
	for oc := 0; oc < c.outChannels; oc++ {
		for t := 0; t < outLen; t++ {
			out[oc*outLen+t] += c.bias[oc]
		}
	}
	return out, outLen, nil
}
