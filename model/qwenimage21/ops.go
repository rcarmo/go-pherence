package qwenimage21

import (
	"fmt"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"math"
)

func ZeroCenteredRMSNorm(dst, x, weight []float32, eps float32) error {
	if len(dst) == 0 || len(dst) != len(x) || len(weight) != len(x) || eps <= 0 {
		return fmt.Errorf("qwen-image-2.1: RMSNorm shape")
	}
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := float32(1 / math.Sqrt(ss/float64(len(x))+float64(eps)))
	for i := range dst {
		dst[i] = x[i] * inv * (1 + weight[i])
	}
	return nil
}
func LayerNorm(dst, x []float32, eps float32) error {
	if len(dst) == 0 || len(dst) != len(x) || eps <= 0 {
		return fmt.Errorf("qwen-image-2.1: LayerNorm shape")
	}
	var mean float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(len(x))
	var variance float64
	for _, v := range x {
		d := float64(v) - mean
		variance += d * d
	}
	inv := float32(1 / math.Sqrt(variance/float64(len(x))+float64(eps)))
	for i, v := range x {
		dst[i] = (v - float32(mean)) * inv
	}
	return nil
}
func TextProjection(dst, x, norm, inWeight, outWeight []float32, inDim, hidden int) error {
	if len(x)%inDim != 0 || len(dst) != (len(x)/inDim)*hidden || len(norm) != inDim || len(inWeight) != hidden*inDim || len(outWeight) != hidden*hidden {
		return fmt.Errorf("qwen-image-2.1: text projection shape")
	}
	tokens := len(x) / inDim
	n := make([]float32, len(x))
	mid := make([]float32, tokens*hidden)
	for t := 0; t < tokens; t++ {
		if e := ZeroCenteredRMSNorm(n[t*inDim:(t+1)*inDim], x[t*inDim:(t+1)*inDim], norm, 1e-6); e != nil {
			return e
		}
	}
	if !simd.MatMul(mid, n, inWeight, tokens, hidden, inDim, false, true) {
		return fmt.Errorf("qwen-image-2.1: text input projection")
	}
	if !simd.GELUTanhTo(mid, mid) {
		return fmt.Errorf("qwen-image-2.1: text GELU")
	}
	if !simd.MatMul(dst, mid, outWeight, tokens, hidden, hidden, false, true) {
		return fmt.Errorf("qwen-image-2.1: text output projection")
	}
	return nil
}
func Modulate(dst, x, params []float32, prefix int, gate bool) error {
	if len(dst) != len(x) || prefix < 0 {
		return fmt.Errorf("qwen-image-2.1: modulation shape")
	}
	if len(params)%2 != 0 {
		return fmt.Errorf("qwen-image-2.1: modulation params")
	}
	hidden := len(params) / 2
	if hidden == 0 || len(x)%hidden != 0 || prefix > len(x)/hidden {
		return fmt.Errorf("qwen-image-2.1: modulation hidden")
	}
	for t := 0; t < len(x)/hidden; t++ {
		row := 0
		if t >= prefix {
			row = 1
		}
		for i := 0; i < hidden; i++ {
			p := params[row*hidden+i]
			if gate {
				p = float32(math.Tanh(float64(p)))
			} else {
				p = 1 + p
			}
			dst[t*hidden+i] = x[t*hidden+i] * p
		}
	}
	return nil
}
func TimestepEmbedding(t float32, dim int) ([]float32, error) {
	if dim <= 0 || dim%2 != 0 {
		return nil, fmt.Errorf("qwen-image-2.1: invalid timestep dim")
	}
	half := dim / 2
	out := make([]float32, dim)
	for i := 0; i < half; i++ {
		freq := math.Exp(-math.Log(10000) * float64(i) / float64(half))
		angle := float64(t) * 1000 * freq
		out[i] = float32(math.Cos(angle))
		out[half+i] = float32(math.Sin(angle))
	}
	return out, nil
}
