package hunyuan3d

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// LinearFloat32 computes y = x @ weight^T + bias for row-major matrices.
// x is [rows,inDim], weight is [outDim,inDim], bias is optional [outDim], y is [rows,outDim].
func LinearFloat32(dst, x, weight, bias []float32, rows, inDim, outDim int) error {
	if rows < 0 || inDim <= 0 || outDim <= 0 {
		return fmt.Errorf("hunyuan3d linear: invalid dims rows=%d in=%d out=%d", rows, inDim, outDim)
	}
	if len(dst) < rows*outDim || len(x) < rows*inDim || len(weight) < outDim*inDim {
		return fmt.Errorf("hunyuan3d linear: short buffer dst=%d x=%d weight=%d", len(dst), len(x), len(weight))
	}
	if bias != nil && len(bias) < outDim {
		return fmt.Errorf("hunyuan3d linear: short bias %d want %d", len(bias), outDim)
	}
	if rows == 0 {
		return nil
	}
	if !simd.DenseNTTo(dst[:rows*outDim], x[:rows*inDim], weight[:outDim*inDim], rows, outDim, inDim, 1, inDim, inDim, outDim) {
		return fmt.Errorf("hunyuan3d linear: SIMD SGEMM rejected validated tensors")
	}
	if bias != nil {
		if !simd.AddBiasRowsTo(dst[:rows*outDim], bias[:outDim], rows, outDim) {
			return fmt.Errorf("hunyuan3d linear: SIMD bias rejected validated tensors")
		}
	}
	return nil
}

func RMSNormFloat32(dst, x, weight []float32, rows, dim int, eps float32) error {
	if rows < 0 || dim <= 0 || len(dst) < rows*dim || len(x) < rows*dim || len(weight) < dim {
		return fmt.Errorf("hunyuan3d rmsnorm: invalid buffers/dims")
	}
	for r := 0; r < rows; r++ {
		row := dst[r*dim : (r+1)*dim]
		copy(row, x[r*dim:(r+1)*dim])
		simd.RMSNorm(row, weight[:dim], eps)
	}
	return nil
}

func GELUTanhInPlace(x []float32) {
	for i, v := range x {
		x[i] = simd.GELUTanhScalar(v)
	}
}

// PatchEmbedFloat32 applies a ViT-style Conv2D patch projection.
// image is CHW [channels,height,width]; weight is [embedDim,channels,patch,patch].
// dst is row-major [numPatches,embedDim].
func PatchEmbedFloat32(dst, image, weight, bias []float32, channels, height, width, patch, embedDim int) (int, error) {
	if channels <= 0 || height <= 0 || width <= 0 || patch <= 0 || embedDim <= 0 || height%patch != 0 || width%patch != 0 {
		return 0, fmt.Errorf("hunyuan3d patch embed: invalid dims c=%d h=%d w=%d patch=%d embed=%d", channels, height, width, patch, embedDim)
	}
	gridH, gridW := height/patch, width/patch
	numPatches := gridH * gridW
	if len(dst) < numPatches*embedDim || len(image) < channels*height*width || len(weight) < embedDim*channels*patch*patch {
		return 0, fmt.Errorf("hunyuan3d patch embed: short buffer")
	}
	if bias != nil && len(bias) < embedDim {
		return 0, fmt.Errorf("hunyuan3d patch embed: short bias")
	}
	for gy := 0; gy < gridH; gy++ {
		for gx := 0; gx < gridW; gx++ {
			pidx := gy*gridW + gx
			out := dst[pidx*embedDim : (pidx+1)*embedDim]
			for e := 0; e < embedDim; e++ {
				sum := float32(0)
				if bias != nil {
					sum = bias[e]
				}
				baseW := e * channels * patch * patch
				for c := 0; c < channels; c++ {
					for py := 0; py < patch; py++ {
						imgOff := c*height*width + (gy*patch+py)*width + gx*patch
						wOff := baseW + c*patch*patch + py*patch
						for px := 0; px < patch; px++ {
							sum += image[imgOff+px] * weight[wOff+px]
						}
					}
				}
				out[e] = sum
			}
		}
	}
	return numPatches, nil
}

// AttentionFloat32 computes multi-head self/cross attention for already-projected Q/K/V.
// q=[qTokens,heads,headDim], k/v=[kvTokens,heads,headDim], dst=[qTokens,heads,headDim].
func AttentionFloat32(dst, q, k, v []float32, qTokens, kvTokens, heads, headDim int, scale float32) error {
	if qTokens < 0 || kvTokens <= 0 || heads <= 0 || headDim <= 0 {
		return fmt.Errorf("hunyuan3d attention: invalid dims")
	}
	needQ := qTokens * heads * headDim
	needKV := kvTokens * heads * headDim
	if len(dst) < needQ || len(q) < needQ || len(k) < needKV || len(v) < needKV {
		return fmt.Errorf("hunyuan3d attention: short buffer")
	}
	if scale == 0 {
		scale = 1 / float32(math.Sqrt(float64(headDim)))
	}
	scores := make([]float32, kvTokens)
	for tq := 0; tq < qTokens; tq++ {
		for h := 0; h < heads; h++ {
			qv := q[(tq*heads+h)*headDim : (tq*heads+h+1)*headDim]
			for tk := 0; tk < kvTokens; tk++ {
				kv := k[(tk*heads+h)*headDim : (tk*heads+h+1)*headDim]
				scores[tk] = simd.Sdot(qv, kv) * scale
			}
			softmaxInPlace(scores)
			out := dst[(tq*heads+h)*headDim : (tq*heads+h+1)*headDim]
			for i := range out {
				out[i] = 0
			}
			for tk, s := range scores {
				vv := v[(tk*heads+h)*headDim : (tk*heads+h+1)*headDim]
				for d := 0; d < headDim; d++ {
					out[d] += s * vv[d]
				}
			}
		}
	}
	return nil
}

func softmaxInPlace(x []float32) {
	maxV := float32(math.Inf(-1))
	for _, v := range x {
		if v > maxV {
			maxV = v
		}
	}
	var sum float32
	for i, v := range x {
		e := float32(math.Exp(float64(v - maxV)))
		x[i] = e
		sum += e
	}
	if sum == 0 {
		return
	}
	inv := 1 / sum
	for i := range x {
		x[i] *= inv
	}
}
