package bert

import (
	"math"

	"github.com/rcarmo/go-pherence/backends/simd/runtime"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
	blasGonum "gonum.org/v1/gonum/blas/gonum"
)

// ForwardFast runs the model with zero allocations in the hot path.
// Uses pre-allocated workspace buffers and SIMD kernels directly.
func init() {
	blas32.Use(blasGonum.Implementation{})
}

func (m *BertModel) ForwardFast(tokenIDs []int, ws *Workspace) []float32 {
	cfg := m.Config
	seqLen := len(tokenIDs)
	h := cfg.HiddenSize
	heads := cfg.NumHeads
	headDim := h / heads

	hidden := ws.buf0[:seqLen*h]

	// Embeddings: word + position + token_type
	wData := m.WordEmb.Data()
	pData := m.PosEmb.Data()
	tData := m.TypeEmb.Data()
	for s := 0; s < seqLen; s++ {
		off := s * h
		wOff := tokenIDs[s] * h
		pOff := s * h
		for d := 0; d < h; d++ {
			hidden[off+d] = wData[wOff+d] + pData[pOff+d] + tData[d] // type 0
		}
	}

	// Embedding LayerNorm
	layerNormInPlace(hidden, m.EmbLnW.Data(), m.EmbLnB.Data(), seqLen, h, 1e-12)

	// Transformer layers
	for l := 0; l < cfg.NumLayers; l++ {
		layer := &m.Layers[l]
		temp := ws.buf1[:seqLen*h]

		// Fused QKV: [seqLen, h] @ [3h, h]^T → [seqLen, 3h]
		// Uses NT matmul so output has contiguous Q,K,V per row
		qkv := ws.qkvBuf[:seqLen*h*3]
		for i := range qkv {
			qkv[i] = 0
		}
		qkvW := layer.QKVW.Data()
		qkvB := layer.QKVB.Data()
		if seqLen > 0 && !simd.SgemmNTTo(qkv, hidden, qkvW, seqLen, 3*h, h, 1.0, h, h, 3*h) {
			return nil
		}
		// Add bias
		for s := 0; s < seqLen; s++ {
			for d := 0; d < 3*h; d++ {
				qkv[s*3*h+d] += qkvB[d]
			}
		}

		// Split Q,K,V: each row has [q(h), k(h), v(h)]
		q := ws.buf1[:seqLen*h]
		kBuf := ws.tempHidden[:seqLen*h]
		vBuf := ws.attnOut[:seqLen*h]
		for s := 0; s < seqLen; s++ {
			copy(q[s*h:(s+1)*h], qkv[s*3*h:s*3*h+h])
			copy(kBuf[s*h:(s+1)*h], qkv[s*3*h+h:s*3*h+2*h])
			copy(vBuf[s*h:(s+1)*h], qkv[s*3*h+2*h:s*3*h+3*h])
		}

		// Multi-head attention
		attnOut := ws.qkvBuf[:seqLen*h] // reuse qkvBuf since QKV split is done
		mhaInPlace(attnOut, q, kBuf, vBuf, ws.scores, seqLen, heads, headDim)

		// Output projection + residual + layernorm
		linearInPlace(temp, attnOut, layer.AttnOutW.Data(), layer.AttnOutB.Data(), seqLen, h, h)
		residualLayerNormInPlace(hidden, temp, layer.AttnLnW.Data(), layer.AttnLnB.Data(), seqLen, h, 1e-12)

		// FFN up: [seqLen, h] @ [h, inter] → [seqLen, inter]
		ffn := ws.ffnBuf[:seqLen*cfg.Intermediate]
		linearInPlace(ffn, hidden, layer.FfnInterW.Data(), layer.FfnInterB.Data(), seqLen, h, cfg.Intermediate)

		// GELU in-place
		geluInPlace(ffn)

		// FFN down: [seqLen, inter] @ [inter, h] → [seqLen, h]
		linearInPlace(temp, ffn, layer.FfnOutW.Data(), layer.FfnOutB.Data(), seqLen, cfg.Intermediate, h)
		residualLayerNormInPlace(hidden, temp, layer.FfnLnW.Data(), layer.FfnLnB.Data(), seqLen, h, 1e-12)
	}

	return hidden
}

// EmbedFast produces an L2-normalized embedding using pre-allocated workspace.
// Call InitWorkspace(maxSeqLen) once before first use.
func (m *BertModel) EmbedFast(tokenIDs []int, attnMask []bool) []float32 {
	ws := m.ws
	if ws == nil || ws.seqLen < len(tokenIDs) {
		ws = newWorkspace(len(tokenIDs), m.Config)
		m.ws = ws
	}
	hidden := m.ForwardFast(tokenIDs, ws)
	h := m.Config.HiddenSize
	seqLen := len(tokenIDs)

	out := ws.outEmb[:h]
	for i := range out {
		out[i] = 0
	}
	count := 0
	for s := 0; s < seqLen; s++ {
		if attnMask[s] {
			for d := 0; d < h; d++ {
				out[d] += hidden[s*h+d]
			}
			count++
		}
	}
	if count > 0 {
		inv := 1.0 / float32(count)
		for d := range out {
			out[d] *= inv
		}
	}
	norm := float32(0)
	for _, v := range out {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		inv := 1.0 / norm
		for i := range out {
			out[i] *= inv
		}
	}
	return out
}

// --- In-place kernels ---

// linearInPlace: out = x @ wT + bias (wT is pre-transposed [inDim, outDim])
func linearInPlace(out, x, wT, bias []float32, m, inDim, outDim int) {
	for i := range out[:m*outDim] {
		out[i] = 0
	}
	// Use gonum BLAS for NN matmul (matches gte-go's gonum path for best cache behavior)
	blas32.Implementation().Sgemm(
		blas.NoTrans, blas.NoTrans,
		m, outDim, inDim,
		1.0,
		x, inDim,
		wT, outDim,
		0.0,
		out, outDim,
	)
	if bias != nil {
		for i := 0; i < m; i++ {
			for j := 0; j < outDim; j++ {
				out[i*outDim+j] += bias[j]
			}
		}
	}
}

func layerNormInPlace(x, gamma, beta []float32, seqLen, h int, eps float32) {
	_ = simd.LayerNormLastAxisTo(x, x, seqLen, h, gamma, beta, eps)
}

func residualLayerNormInPlace(residual, x, gamma, beta []float32, seqLen, h int, eps float32) {
	total := seqLen * h
	if total <= 0 || len(residual) < total || len(x) < total {
		return
	}
	_ = simd.VecAddTo(x[:total], x[:total], residual[:total])
	_ = simd.LayerNormLastAxisTo(residual[:total], x[:total], seqLen, h, gamma, beta, eps)
}

func geluInPlace(x []float32) {
	_ = simd.GELUTanhTo(x, x)
}

func mhaInPlace(out, q, k, v, scoresBuf []float32, seqLen, heads, headDim int) {
	hidden := heads * headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	for h := 0; h < heads; h++ {
		scores := scoresBuf[h*seqLen*seqLen : (h+1)*seqLen*seqLen]

		// Q·K^T per head
		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				sum := float32(0)
				for d := 0; d < headDim; d++ {
					sum += q[i*hidden+h*headDim+d] * k[j*hidden+h*headDim+d]
				}
				scores[i*seqLen+j] = sum * scale
			}
		}

		// Softmax per row
		simd.SoftmaxRowsInPlace(scores, seqLen, seqLen)

		// Context: scores @ V per head
		for i := 0; i < seqLen; i++ {
			for d := 0; d < headDim; d++ {
				sum := float32(0)
				for j := 0; j < seqLen; j++ {
					sum += scores[i*seqLen+j] * v[j*hidden+h*headDim+d]
				}
				out[i*hidden+h*headDim+d] = sum
			}
		}
	}
}
