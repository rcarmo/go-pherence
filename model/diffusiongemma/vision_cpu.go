package diffusiongemma

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// VisionLayerF32 is a small reference-style CPU executor input used to validate
// the semantic vision tower operation order independently from checkpoint IO.
// Weights are row-major [rows, cols]. It is intentionally not wired into default
// generation until full image reference fixtures exist.
type VisionLayerF32 struct {
	InputLayerNorm         []float32
	PostAttentionLayerNorm []float32
	PreFFNLayerNorm        []float32
	PostFFNLayerNorm       []float32
	QProj                  []float32
	KProj                  []float32
	VProj                  []float32
	OProj                  []float32
	QNorm                  []float32
	KNorm                  []float32
	MLPGateProj            []float32
	MLPUpProj              []float32
	MLPDownProj            []float32
	MLPIntermediate        int
	PatchWidth             int
	PatchHeight            int
	RoPETheta              float32
}

func RunVisionLayerF32(hidden []float32, seqLen, hiddenSize, heads, headDim int, layer VisionLayerF32) error {
	if seqLen <= 0 || hiddenSize <= 0 || heads <= 0 || headDim <= 0 || len(hidden) != seqLen*hiddenSize {
		return fmt.Errorf("DiffusionGemma vision layer invalid hidden shape len=%d seq=%d hidden=%d heads=%d head_dim=%d", len(hidden), seqLen, hiddenSize, heads, headDim)
	}
	if heads*headDim != hiddenSize {
		return fmt.Errorf("DiffusionGemma vision layer heads*head_dim=%d want hidden=%d", heads*headDim, hiddenSize)
	}
	if len(layer.InputLayerNorm) != hiddenSize || len(layer.PostAttentionLayerNorm) != hiddenSize || len(layer.PreFFNLayerNorm) != hiddenSize || len(layer.PostFFNLayerNorm) != hiddenSize {
		return fmt.Errorf("DiffusionGemma vision layer norm shape mismatch")
	}
	if !visionMatrixShape(layer.QProj, hiddenSize, hiddenSize) || !visionMatrixShape(layer.KProj, hiddenSize, hiddenSize) || !visionMatrixShape(layer.VProj, hiddenSize, hiddenSize) || !visionMatrixShape(layer.OProj, hiddenSize, hiddenSize) {
		return fmt.Errorf("DiffusionGemma vision layer attention projection shape mismatch")
	}
	if len(layer.QNorm) != headDim || len(layer.KNorm) != headDim {
		return fmt.Errorf("DiffusionGemma vision layer q/k norm len=%d/%d want %d", len(layer.QNorm), len(layer.KNorm), headDim)
	}
	inter := layer.MLPIntermediate
	if inter <= 0 || !visionMatrixShape(layer.MLPGateProj, inter, hiddenSize) || !visionMatrixShape(layer.MLPUpProj, inter, hiddenSize) || !visionMatrixShape(layer.MLPDownProj, hiddenSize, inter) {
		return fmt.Errorf("DiffusionGemma vision layer MLP projection shape mismatch")
	}

	residual := append([]float32(nil), hidden...)
	normed := append([]float32(nil), hidden...)
	for pos := 0; pos < seqLen; pos++ {
		row := normed[pos*hiddenSize : (pos+1)*hiddenSize]
		if !simd.RMSNormTo(row, layer.InputLayerNorm, 1e-6) {
			return fmt.Errorf("DiffusionGemma vision input norm rejected row %d", pos)
		}
	}
	q := make([]float32, seqLen*hiddenSize)
	k := make([]float32, seqLen*hiddenSize)
	v := make([]float32, seqLen*hiddenSize)
	for pos := 0; pos < seqLen; pos++ {
		row := normed[pos*hiddenSize : (pos+1)*hiddenSize]
		if !simd.GemvRows(q[pos*hiddenSize:(pos+1)*hiddenSize], row, layer.QProj, hiddenSize, hiddenSize) || !simd.GemvRows(k[pos*hiddenSize:(pos+1)*hiddenSize], row, layer.KProj, hiddenSize, hiddenSize) || !simd.GemvRows(v[pos*hiddenSize:(pos+1)*hiddenSize], row, layer.VProj, hiddenSize, hiddenSize) {
			return fmt.Errorf("DiffusionGemma vision attention projection rejected row %d", pos)
		}
		for h := 0; h < heads; h++ {
			qh := q[pos*hiddenSize+h*headDim : pos*hiddenSize+(h+1)*headDim]
			kh := k[pos*hiddenSize+h*headDim : pos*hiddenSize+(h+1)*headDim]
			if !simd.RMSNormTo(qh, layer.QNorm, 1e-6) || !simd.RMSNormTo(kh, layer.KNorm, 1e-6) {
				return fmt.Errorf("DiffusionGemma vision q/k norm rejected row %d head %d", pos, h)
			}
			if layer.PatchWidth > 0 {
				if layer.PatchHeight <= 0 || layer.PatchWidth*layer.PatchHeight != seqLen {
					return fmt.Errorf("DiffusionGemma vision RoPE grid=%dx%d does not match seq=%d", layer.PatchWidth, layer.PatchHeight, seqLen)
				}
				theta := layer.RoPETheta
				if theta == 0 {
					theta = 10000
				}
				if !applyVisionRoPE2D(qh, headDim, pos%layer.PatchWidth, pos/layer.PatchWidth, theta) || !applyVisionRoPE2D(kh, headDim, pos%layer.PatchWidth, pos/layer.PatchWidth, theta) {
					return fmt.Errorf("DiffusionGemma vision RoPE rejected head_dim=%d", headDim)
				}
			}
			if !simd.RMSNormNoScaleTo(v[pos*hiddenSize+h*headDim:pos*hiddenSize+(h+1)*headDim], 1e-6) {
				return fmt.Errorf("DiffusionGemma vision v norm rejected row %d head %d", pos, h)
			}
		}
	}
	attnOut := make([]float32, seqLen*hiddenSize)
	scores := make([]float32, seqLen)
	for pos := 0; pos < seqLen; pos++ {
		for h := 0; h < heads; h++ {
			qv := q[pos*hiddenSize+h*headDim : pos*hiddenSize+(h+1)*headDim]
			maxScore := float32(math.Inf(-1))
			for j := 0; j < seqLen; j++ {
				kv := k[j*hiddenSize+h*headDim : j*hiddenSize+(h+1)*headDim]
				s := visionDot(qv, kv)
				scores[j] = s
				if s > maxScore {
					maxScore = s
				}
			}
			var sum float64
			for j := 0; j < seqLen; j++ {
				e := math.Exp(float64(scores[j] - maxScore))
				sum += e
				scores[j] = float32(e)
			}
			out := attnOut[pos*hiddenSize+h*headDim : pos*hiddenSize+(h+1)*headDim]
			for j := 0; j < seqLen; j++ {
				p := scores[j] / float32(sum)
				vv := v[j*hiddenSize+h*headDim : j*hiddenSize+(h+1)*headDim]
				for d := 0; d < headDim; d++ {
					out[d] += p * vv[d]
				}
			}
		}
	}
	projAttn := make([]float32, hiddenSize)
	for pos := 0; pos < seqLen; pos++ {
		if !simd.GemvRows(projAttn, attnOut[pos*hiddenSize:(pos+1)*hiddenSize], layer.OProj, hiddenSize, hiddenSize) {
			return fmt.Errorf("DiffusionGemma vision o_proj rejected row %d", pos)
		}
		if !simd.RMSNormTo(projAttn, layer.PostAttentionLayerNorm, 1e-6) {
			return fmt.Errorf("DiffusionGemma vision post-attention norm rejected row %d", pos)
		}
		row := hidden[pos*hiddenSize : (pos+1)*hiddenSize]
		for i := 0; i < hiddenSize; i++ {
			row[i] = residual[pos*hiddenSize+i] + projAttn[i]
		}
	}

	residual = append(residual[:0], hidden...)
	gate := make([]float32, inter)
	up := make([]float32, inter)
	act := make([]float32, inter)
	down := make([]float32, hiddenSize)
	for pos := 0; pos < seqLen; pos++ {
		row := hidden[pos*hiddenSize : (pos+1)*hiddenSize]
		ffnIn := append([]float32(nil), row...)
		if !simd.RMSNormTo(ffnIn, layer.PreFFNLayerNorm, 1e-6) {
			return fmt.Errorf("DiffusionGemma vision pre-FFN norm rejected row %d", pos)
		}
		if !simd.GemvRows(gate, ffnIn, layer.MLPGateProj, inter, hiddenSize) || !simd.GemvRows(up, ffnIn, layer.MLPUpProj, inter, hiddenSize) {
			return fmt.Errorf("DiffusionGemma vision MLP gate/up rejected row %d", pos)
		}
		if !simd.GELUExactMulTo(act, gate, up) {
			return fmt.Errorf("DiffusionGemma vision MLP activation rejected row %d", pos)
		}
		if !simd.GemvRows(down, act, layer.MLPDownProj, hiddenSize, inter) {
			return fmt.Errorf("DiffusionGemma vision MLP down rejected row %d", pos)
		}
		if !simd.RMSNormTo(down, layer.PostFFNLayerNorm, 1e-6) {
			return fmt.Errorf("DiffusionGemma vision post-FFN norm rejected row %d", pos)
		}
		for i := 0; i < hiddenSize; i++ {
			row[i] = residual[pos*hiddenSize+i] + down[i]
		}
	}
	return nil
}

func applyVisionRoPE2D(x []float32, headDim, posX, posY int, theta float32) bool {
	if len(x) != headDim || headDim <= 0 || headDim%4 != 0 || theta <= 0 {
		return false
	}
	spatialDim := headDim / 2
	half := spatialDim / 2
	positions := [2]int{posX, posY}
	for axis, position := range positions {
		offset := axis * spatialDim
		for i := 0; i < half; i++ {
			angle := float64(position) / math.Pow(float64(theta), float64(2*i)/float64(spatialDim))
			c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
			a, b := x[offset+i], x[offset+half+i]
			x[offset+i] = a*c - b*s
			x[offset+half+i] = b*c + a*s
		}
	}
	return true
}

func visionMatrixShape(w []float32, rows, cols int) bool {
	return rows > 0 && cols > 0 && len(w) == rows*cols
}

func visionDot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
