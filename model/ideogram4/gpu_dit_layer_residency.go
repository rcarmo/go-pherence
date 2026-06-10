package ideogram4

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

type ditLayerGPUResidency struct {
	qkv   *nvidia.GPUFP8E4M3Linear
	o     *nvidia.GPUFP8E4M3Linear
	w1    *nvidia.GPUFP8E4M3Linear
	w2    *nvidia.GPUFP8E4M3Linear
	w3    *nvidia.GPUFP8E4M3Linear
	adaln *nvidia.GPUFP8E4M3Linear

	attnN1 *nvidia.Buffer
	attnN2 *nvidia.Buffer
	ffnN1  *nvidia.Buffer
	ffnN2  *nvidia.Buffer
	normQ  *nvidia.Buffer
	normK  *nvidia.Buffer
}

func gpuFullLayerIslandEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_FULL_LAYER")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func (l *DiTLayer) cacheGPUResidency() bool {
	if l == nil || !gpuFP8CacheEnabled() {
		return false
	}
	winS := strings.TrimSpace(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_WINDOW"))
	if winS == "" || winS == "0" {
		return false
	}
	win, err := strconv.Atoi(winS)
	if err != nil || win <= 0 {
		return false
	}
	start, _ := strconv.Atoi(strings.TrimSpace(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_START")))
	return l.Index >= start && l.Index < start+win
}

func (l *DiTLayer) uploadGPU() (*ditLayerGPUResidency, error) {
	if l != nil && l.cacheGPUResidency() && l.gpu != nil {
		return l.gpu, nil
	}
	r, err := uploadDiTLayerGPU(*l)
	if err != nil {
		return nil, err
	}
	if l != nil && l.cacheGPUResidency() {
		l.gpu = r
	}
	return r, nil
}

func uploadDiTLayerGPU(l DiTLayer) (*ditLayerGPUResidency, error) {
	if !gpuFP8Enabled() {
		return nil, nil
	}
	if !nvidia.Available() {
		return nil, fmt.Errorf("nvidia runtime unavailable")
	}
	r := &ditLayerGPUResidency{}
	upload := func(dst **nvidia.GPUFP8E4M3Linear, name string, lin *FP8Linear) error {
		if lin == nil {
			return fmt.Errorf("nil %s linear", name)
		}
		w, err := nvidia.UploadFP8E4M3Linear(lin.weight.Weight, lin.weight.Scale, lin.weight.Bias, lin.weight.OutDim, lin.weight.InDim)
		if err != nil {
			return fmt.Errorf("upload %s: %w", name, err)
		}
		*dst = w
		return nil
	}
	if err := upload(&r.qkv, "qkv", l.QKV); err != nil {
		r.Free()
		return nil, err
	}
	if err := upload(&r.o, "o", l.O); err != nil {
		r.Free()
		return nil, err
	}
	if err := upload(&r.w1, "w1", l.W1); err != nil {
		r.Free()
		return nil, err
	}
	if err := upload(&r.w2, "w2", l.W2); err != nil {
		r.Free()
		return nil, err
	}
	if err := upload(&r.w3, "w3", l.W3); err != nil {
		r.Free()
		return nil, err
	}
	// Keep per-layer AdaLN modulation on the CPU path for now. In full DiT
	// residency, the resident AdaLN GEMV diverges in-context after the first
	// layers even though standalone FP8 GEMV comparisons pass; QKV/O/MLP GEMMs
	// remain GPU-resident and compare cleanly. This preserves correctness while
	// the AdaLN-specific residency interaction is debugged.
	uploadF32 := func(dst **nvidia.Buffer, name string, data []float32) error {
		buf, err := nvidia.Malloc(len(data))
		if err != nil {
			return fmt.Errorf("upload %s metadata alloc: %w", name, err)
		}
		if err := buf.Upload(data); err != nil {
			buf.Free()
			return fmt.Errorf("upload %s metadata: %w", name, err)
		}
		*dst = buf
		return nil
	}
	for _, item := range []struct {
		dst  **nvidia.Buffer
		name string
		data []float32
	}{
		{&r.attnN1, "attnN1", l.AttnN1},
		{&r.attnN2, "attnN2", l.AttnN2},
		{&r.ffnN1, "ffnN1", l.FfnN1},
		{&r.ffnN2, "ffnN2", l.FfnN2},
		{&r.normQ, "normQ", l.NormQ},
		{&r.normK, "normK", l.NormK},
	} {
		if err := uploadF32(item.dst, item.name, item.data); err != nil {
			r.Free()
			return nil, err
		}
	}
	return r, nil
}

func (r *ditLayerGPUResidency) Free() {
	if r == nil {
		return
	}
	for _, w := range []*nvidia.GPUFP8E4M3Linear{r.qkv, r.o, r.w1, r.w2, r.w3, r.adaln} {
		w.Free()
	}
	for _, b := range []*nvidia.Buffer{r.attnN1, r.attnN2, r.ffnN1, r.ffnN2, r.normQ, r.normK} {
		b.Free()
	}
	r.qkv, r.o, r.w1, r.w2, r.w3, r.adaln = nil, nil, nil, nil, nil, nil
	r.attnN1, r.attnN2, r.ffnN1, r.ffnN2, r.normQ, r.normK = nil, nil, nil, nil, nil, nil
}

func (r *ditLayerGPUResidency) gemv(name string, gpuW *nvidia.GPUFP8E4M3Linear, cpuW *FP8Linear, x, out []float32) error {
	if r != nil && gpuW != nil {
		if err := nvidia.GemvFP8E4M3(out, x, gpuW); err == nil {
			return nil
		} else if gpuFP8Strict() {
			return fmt.Errorf("DiT layer GPU %s GEMV: %w", name, err)
		}
	}
	return cpuW.weight.GemvTo(x, out)
}

func (r *ditLayerGPUResidency) QKV(l DiTLayer, x, out []float32) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.qkv
	}
	return r.gemv("qkv", w, l.QKV, x, out)
}
func (r *ditLayerGPUResidency) O(l DiTLayer, x, out []float32) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.o
	}
	return r.gemv("o", w, l.O, x, out)
}
func (r *ditLayerGPUResidency) W1(l DiTLayer, x, out []float32) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.w1
	}
	return r.gemv("w1", w, l.W1, x, out)
}
func (r *ditLayerGPUResidency) W2(l DiTLayer, x, out []float32) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.w2
	}
	return r.gemv("w2", w, l.W2, x, out)
}
func (r *ditLayerGPUResidency) W3(l DiTLayer, x, out []float32) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.w3
	}
	return r.gemv("w3", w, l.W3, x, out)
}
func (r *ditLayerGPUResidency) AdaLN(l DiTLayer, x, out []float32) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.adaln
	}
	return r.gemv("adaln", w, l.AdaLN, x, out)
}

func (r *ditLayerGPUResidency) gemm(name string, gpuW *nvidia.GPUFP8E4M3Linear, cpuW *FP8Linear, x, out []float32, batch int) error {
	if r != nil && gpuW != nil {
		if err := nvidia.GemmFP8E4M3(out, x, batch, gpuW); err == nil {
			return nil
		} else if gpuFP8Strict() {
			return fmt.Errorf("DiT layer GPU %s GEMM: %w", name, err)
		}
	}
	return cpuW.ApplyBatch(x, out, batch)
}

func (r *ditLayerGPUResidency) FullLayerIslandsBuffer(l DiTLayer, hiddenBuf, normedBuf, cosBuf, sinBuf *nvidia.Buffer, scaleMSA, gateMSA, scaleMLP, gateMLP []float32, tokens, heads, headDim int, scaleAttn, normEps float32) error {
	if r == nil || r.qkv == nil || r.o == nil || r.w1 == nil || r.w3 == nil || r.w2 == nil || cosBuf == nil || sinBuf == nil {
		return fmt.Errorf("full layer island unavailable")
	}
	emb := heads * headDim
	alloc := func(name string, size int) (*nvidia.Buffer, error) {
		b, err := nvidia.Malloc(size)
		if err != nil {
			return nil, fmt.Errorf("alloc DiT full-layer buffer %s: %w", name, err)
		}
		return b, nil
	}
	scaleMSABuf, err := alloc("scaleMSA", emb)
	if err != nil {
		return err
	}
	defer scaleMSABuf.Free()
	gateMSABuf, err := alloc("gateMSA", emb)
	if err != nil {
		return err
	}
	defer gateMSABuf.Free()
	scaleMLPBuf, err := alloc("scaleMLP", emb)
	if err != nil {
		return err
	}
	defer scaleMLPBuf.Free()
	gateMLPBuf, err := alloc("gateMLP", emb)
	if err != nil {
		return err
	}
	defer gateMLPBuf.Free()
	if err := scaleMSABuf.Upload(scaleMSA[:emb]); err != nil {
		return err
	}
	if err := gateMSABuf.Upload(gateMSA[:emb]); err != nil {
		return err
	}
	if err := scaleMLPBuf.Upload(scaleMLP[:emb]); err != nil {
		return err
	}
	if err := gateMLPBuf.Upload(gateMLP[:emb]); err != nil {
		return err
	}
	attnN1, attnN2, ffnN1, ffnN2, normQ, normK := r.attnN1, r.attnN2, r.ffnN1, r.ffnN2, r.normQ, r.normK
	var owned []*nvidia.Buffer
	upload := func(data []float32) (*nvidia.Buffer, error) {
		b, err := nvidia.Malloc(len(data))
		if err != nil {
			return nil, err
		}
		if err := b.Upload(data); err != nil {
			b.Free()
			return nil, err
		}
		owned = append(owned, b)
		return b, nil
	}
	if attnN1 == nil {
		if attnN1, err = upload(l.AttnN1); err != nil {
			return err
		}
	}
	if attnN2 == nil {
		if attnN2, err = upload(l.AttnN2); err != nil {
			return err
		}
	}
	if ffnN1 == nil {
		if ffnN1, err = upload(l.FfnN1); err != nil {
			return err
		}
	}
	if ffnN2 == nil {
		if ffnN2, err = upload(l.FfnN2); err != nil {
			return err
		}
	}
	if normQ == nil {
		if normQ, err = upload(l.NormQ); err != nil {
			return err
		}
	}
	if normK == nil {
		if normK, err = upload(l.NormK); err != nil {
			return err
		}
	}
	defer func() {
		for _, b := range owned {
			b.Free()
		}
	}()
	if err := nvidia.IdeogramRMSNormRowsBuffer(normedBuf, hiddenBuf, attnN1, scaleMSABuf, tokens, emb, normEps, true); err != nil {
		return err
	}
	if err := nvidia.GemmQKVAttentionOResidualFP8E4M3Buffer(hiddenBuf, normedBuf, gateMSABuf, normQ, normK, attnN2, cosBuf, sinBuf, tokens, heads, headDim, scaleAttn, r.qkv, r.o); err != nil {
		return err
	}
	if err := nvidia.IdeogramRMSNormRowsBuffer(normedBuf, hiddenBuf, ffnN1, scaleMLPBuf, tokens, emb, normEps, true); err != nil {
		return err
	}
	return nvidia.GemmSwiGLUResidualFP8E4M3Buffer(hiddenBuf, normedBuf, gateMLPBuf, ffnN2, tokens, r.w1, r.w3, r.w2)
}

func (r *ditLayerGPUResidency) FullLayerIslands(l DiTLayer, hidden, scaleMSA, gateMSA, scaleMLP, gateMLP []float32, tokens, heads, headDim int, rope *MRoPE, scaleAttn, normEps float32) error {
	if r == nil || r.qkv == nil || r.o == nil || r.w1 == nil || r.w3 == nil || r.w2 == nil || rope == nil {
		return fmt.Errorf("full layer island unavailable")
	}
	if r.attnN1 != nil && r.attnN2 != nil && r.ffnN1 != nil && r.ffnN2 != nil && r.normQ != nil && r.normK != nil {
		emb := heads * headDim
		n := tokens * emb
		tableLen := tokens * (headDim / 2)
		alloc := func(name string, size int) (*nvidia.Buffer, error) {
			b, err := nvidia.Malloc(size)
			if err != nil {
				return nil, fmt.Errorf("alloc DiT full-layer cached %s: %w", name, err)
			}
			return b, nil
		}
		hiddenBuf, err := alloc("hidden", n)
		if err != nil {
			return err
		}
		defer hiddenBuf.Free()
		normedBuf, err := alloc("normed", n)
		if err != nil {
			return err
		}
		defer normedBuf.Free()
		scaleMSABuf, err := alloc("scaleMSA", emb)
		if err != nil {
			return err
		}
		defer scaleMSABuf.Free()
		gateMSABuf, err := alloc("gateMSA", emb)
		if err != nil {
			return err
		}
		defer gateMSABuf.Free()
		scaleMLPBuf, err := alloc("scaleMLP", emb)
		if err != nil {
			return err
		}
		defer scaleMLPBuf.Free()
		gateMLPBuf, err := alloc("gateMLP", emb)
		if err != nil {
			return err
		}
		defer gateMLPBuf.Free()
		cosBuf, err := alloc("cos", tableLen)
		if err != nil {
			return err
		}
		defer cosBuf.Free()
		sinBuf, err := alloc("sin", tableLen)
		if err != nil {
			return err
		}
		defer sinBuf.Free()
		if err := hiddenBuf.Upload(hidden[:n]); err != nil {
			return err
		}
		if err := scaleMSABuf.Upload(scaleMSA[:emb]); err != nil {
			return err
		}
		if err := gateMSABuf.Upload(gateMSA[:emb]); err != nil {
			return err
		}
		if err := scaleMLPBuf.Upload(scaleMLP[:emb]); err != nil {
			return err
		}
		if err := gateMLPBuf.Upload(gateMLP[:emb]); err != nil {
			return err
		}
		if err := cosBuf.Upload(rope.cos[:tableLen]); err != nil {
			return err
		}
		if err := sinBuf.Upload(rope.sin[:tableLen]); err != nil {
			return err
		}
		if err := nvidia.IdeogramRMSNormRowsBuffer(normedBuf, hiddenBuf, r.attnN1, scaleMSABuf, tokens, emb, normEps, true); err != nil {
			return err
		}
		if err := nvidia.GemmQKVAttentionOResidualFP8E4M3Buffer(hiddenBuf, normedBuf, gateMSABuf, r.normQ, r.normK, r.attnN2, cosBuf, sinBuf, tokens, heads, headDim, scaleAttn, r.qkv, r.o); err != nil {
			return err
		}
		if err := nvidia.IdeogramRMSNormRowsBuffer(normedBuf, hiddenBuf, r.ffnN1, scaleMLPBuf, tokens, emb, normEps, true); err != nil {
			return err
		}
		if err := nvidia.GemmSwiGLUResidualFP8E4M3Buffer(hiddenBuf, normedBuf, gateMLPBuf, r.ffnN2, tokens, r.w1, r.w3, r.w2); err != nil {
			return err
		}
		return hiddenBuf.Download(hidden[:n])
	}
	if err := nvidia.GemmDiTLayerIslandsFP8E4M3(hidden, l.AttnN1, scaleMSA, gateMSA, l.NormQ, l.NormK, l.AttnN2, l.FfnN1, scaleMLP, gateMLP, l.FfnN2, rope.cos, rope.sin, tokens, heads, headDim, scaleAttn, normEps, r.qkv, r.o, r.w1, r.w3, r.w2); err == nil {
		return nil
	} else if gpuFP8Strict() || gpuAttentionStrict() || gpuNormStrict() || gpuMRoPEStrict() || gpuMLPStrict() {
		return fmt.Errorf("DiT layer GPU full islands: %w", err)
	}
	return fmt.Errorf("full layer island unavailable")
}

func (r *ditLayerGPUResidency) AttentionResidualBatch(l DiTLayer, hidden, x, gate []float32, tokens, heads, headDim int, rope *MRoPE, scale, normEps float32) error {
	if r != nil && r.qkv != nil && r.o != nil && rope != nil {
		if err := nvidia.GemmQKVAttentionOResidualFP8E4M3(hidden, x, gate, l.NormQ, l.NormK, l.AttnN2, rope.cos, rope.sin, tokens, heads, headDim, scale, r.qkv, r.o); err == nil {
			return nil
		} else if gpuFP8Strict() || gpuAttentionStrict() || gpuNormStrict() || gpuMRoPEStrict() {
			return fmt.Errorf("DiT layer GPU QKV+attention+O+residual: %w", err)
		}
	}
	attnOut := make([]float32, tokens*heads*headDim)
	if err := r.QKVAttentionBatch(l, x, attnOut, tokens, heads, headDim, rope, scale); err != nil {
		return err
	}
	oprojAll := make([]float32, len(attnOut))
	if err := r.OBatch(l, attnOut, oprojAll, tokens); err != nil {
		return err
	}
	postNormAll := make([]float32, len(attnOut))
	if err := rmsNormRowsWeightedGPU(postNormAll, oprojAll, l.AttnN2, nil, tokens, heads*headDim, normEps); err != nil {
		return err
	}
	return gatedResidualRowsGPU(hidden, postNormAll, gate, tokens, heads*headDim)
}

func (r *ditLayerGPUResidency) QKVAttentionBatch(l DiTLayer, x, out []float32, tokens, heads, headDim int, rope *MRoPE, scale float32) error {
	if r != nil && r.qkv != nil && rope != nil {
		if err := nvidia.GemmQKVAttentionFP8E4M3(out, x, l.NormQ, l.NormK, rope.cos, rope.sin, tokens, heads, headDim, scale, r.qkv); err == nil {
			return nil
		} else if gpuFP8Strict() || gpuAttentionStrict() || gpuNormStrict() || gpuMRoPEStrict() {
			return fmt.Errorf("DiT layer GPU QKV+attention: %w", err)
		}
	}
	q := make([]float32, tokens*heads*headDim)
	k := make([]float32, tokens*heads*headDim)
	v := make([]float32, tokens*heads*headDim)
	qkvAll := make([]float32, tokens*3*heads*headDim)
	if err := r.QKVBatch(l, x, qkvAll, tokens); err != nil {
		return err
	}
	emb := heads * headDim
	for t := 0; t < tokens; t++ {
		qkv := qkvAll[t*3*emb : (t+1)*3*emb]
		copy(q[t*emb:(t+1)*emb], qkv[0:emb])
		copy(k[t*emb:(t+1)*emb], qkv[emb:2*emb])
		copy(v[t*emb:(t+1)*emb], qkv[2*emb:3*emb])
	}
	if gpuNormEnabled() {
		if err := rmsNormRowsWeightedGPU(q, q, l.NormQ, nil, tokens*heads, headDim, 1e-5); err != nil {
			return err
		}
		if err := rmsNormRowsWeightedGPU(k, k, l.NormK, nil, tokens*heads, headDim, 1e-5); err != nil {
			return err
		}
	} else {
		for t := 0; t < tokens; t++ {
			for h := 0; h < heads; h++ {
				off := t*emb + h*headDim
				rmsNormWeightedCPU(q[off:off+headDim], q[off:off+headDim], l.NormQ, 1e-5)
				rmsNormWeightedCPU(k[off:off+headDim], k[off:off+headDim], l.NormK, 1e-5)
			}
		}
	}
	if err := applyMRoPEToQK(q, k, rope, tokens, heads, headDim); err != nil {
		return err
	}
	return fullSelfAttention(out, q, k, v, tokens, heads, headDim, scale)
}

func (r *ditLayerGPUResidency) QKVBatch(l DiTLayer, x, out []float32, batch int) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.qkv
	}
	return r.gemm("qkv", w, l.QKV, x, out, batch)
}
func (r *ditLayerGPUResidency) OBatch(l DiTLayer, x, out []float32, batch int) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.o
	}
	return r.gemm("o", w, l.O, x, out, batch)
}
func (r *ditLayerGPUResidency) W1Batch(l DiTLayer, x, out []float32, batch int) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.w1
	}
	return r.gemm("w1", w, l.W1, x, out, batch)
}
func (r *ditLayerGPUResidency) W2Batch(l DiTLayer, x, out []float32, batch int) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.w2
	}
	return r.gemm("w2", w, l.W2, x, out, batch)
}
func (r *ditLayerGPUResidency) W3Batch(l DiTLayer, x, out []float32, batch int) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.w3
	}
	return r.gemm("w3", w, l.W3, x, out, batch)
}

func (r *ditLayerGPUResidency) MLPResidualBatch(l DiTLayer, hidden, x, gate []float32, batch int) error {
	if r != nil && r.w1 != nil && r.w3 != nil && r.w2 != nil {
		if err := nvidia.GemmSwiGLUResidualFP8E4M3(hidden, x, gate, l.FfnN2, batch, r.w1, r.w3, r.w2); err == nil {
			return nil
		} else if gpuFP8Strict() || gpuNormStrict() || gpuMLPStrict() {
			return fmt.Errorf("DiT layer GPU SwiGLU+W2+residual: %w", err)
		}
	}
	downAll := make([]float32, batch*l.W2.OutDim())
	if err := r.MLPBatch(l, x, downAll, batch); err != nil {
		return err
	}
	postNormAll := make([]float32, len(downAll))
	if err := rmsNormRowsWeightedGPU(postNormAll, downAll, l.FfnN2, nil, batch, l.W2.OutDim(), 1e-5); err != nil {
		return err
	}
	return gatedResidualRowsGPU(hidden, postNormAll, gate, batch, l.W2.OutDim())
}

func (r *ditLayerGPUResidency) MLPBatch(l DiTLayer, x, out []float32, batch int) error {
	if r != nil && r.w1 != nil && r.w3 != nil && r.w2 != nil {
		if err := nvidia.GemmSwiGLUFP8E4M3(out, x, batch, r.w1, r.w3, r.w2); err == nil {
			return nil
		} else if gpuFP8Strict() {
			return fmt.Errorf("DiT layer GPU SwiGLU+W2: %w", err)
		}
	}
	gAll := make([]float32, batch*l.W1.OutDim())
	uAll := make([]float32, batch*l.W3.OutDim())
	if err := r.W1W3Batch(l, x, gAll, uAll, batch); err != nil {
		return err
	}
	siluMulInPlace(gAll, uAll)
	return r.W2Batch(l, gAll, out, batch)
}

func (r *ditLayerGPUResidency) W1W3Batch(l DiTLayer, x, outW1, outW3 []float32, batch int) error {
	if r != nil && r.w1 != nil && r.w3 != nil {
		if err := nvidia.Gemm2FP8E4M3SameInput(outW1, outW3, x, batch, r.w1, r.w3); err == nil {
			return nil
		} else if gpuFP8Strict() {
			return fmt.Errorf("DiT layer GPU w1+w3 GEMM2: %w", err)
		}
	}
	if err := l.W1.ApplyBatch(x, outW1, batch); err != nil {
		return err
	}
	return l.W3.ApplyBatch(x, outW3, batch)
}
