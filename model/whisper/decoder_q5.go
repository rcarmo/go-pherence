package whisper

import (
	"context"
	"fmt"
	"runtime"
	"sync"

	simdrt "github.com/rcarmo/go-pherence/backends/simd/runtime"
	legacy "github.com/rcarmo/go-pherence/loader/whisperggml"
)

// decoderLayerQ5 holds a decoder layer's original packed Q5_0 projection rows.
type decoderLayerQ5 struct {
	selfQ, selfK, selfV, selfO, crossQ, crossO, fc1, fc2 []byte
}

// attachOriginalQ5 makes the CPU decoder read its projection and LM-head
// weights as the pinned file's packed Q5_0 rows through the fused AVX2 kernel.
// Every packed value is checked bit-equal to the widened F32 weight it replaces,
// and the kernel accumulates in Sdot's order, so logits are bit-identical to
// the F32 decoder while reading ~5.8x fewer weight bytes. Explicit only; there
// is no fallback: unsupported CPUs, int8 or GPU decoder paths are rejected.
// The attached (fast) decoder also splits unobserved cross-attention heads
// across workers; per-head arithmetic is unchanged, so results stay identical.
func (dec *Decoder) attachOriginalQ5(ctx context.Context, file *legacy.File) error {
	if ctx == nil || dec == nil || file == nil {
		return fmt.Errorf("whisper Q5 decoder: nil argument")
	}
	if !simdrt.HasSdotQ5_0Asm {
		return fmt.Errorf("whisper Q5 decoder: fused AVX2/FMA/F16C kernel unavailable")
	}
	if useInt8 || dec.lmHeadGPU != nil {
		return fmt.Errorf("whisper Q5 decoder: conflicts with int8/GPU decoder paths")
	}
	cfg := dec.cfg
	d, f := cfg.DecoderDModel, cfg.DecoderFFNDim
	if d%32 != 0 || f%32 != 0 || len(dec.Layers) != cfg.DecoderLayers {
		return fmt.Errorf("whisper Q5 decoder: geometry")
	}
	load := func(name string, values []float32, out, in int) ([]byte, error) {
		raw, shape, err := file.Q5Blocks(ctx, name)
		if err != nil {
			return nil, err
		}
		if len(shape) != 2 || shape[0] != in || shape[1] != out || len(values) != out*in {
			return nil, fmt.Errorf("whisper Q5 decoder: shape %s", name)
		}
		if err := checkOriginalQ5Values(ctx, raw, values); err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		return raw, nil
	}
	layers := make([]decoderLayerQ5, len(dec.Layers))
	for l := range dec.Layers {
		L := &dec.Layers[l]
		if L.gpuFC1Weight != nil || L.gpuFC2Weight != nil {
			return fmt.Errorf("whisper Q5 decoder: conflicts with GPU FFN")
		}
		p := fmt.Sprintf("decoder.blocks.%d.", l)
		for _, m := range []struct {
			dst     *[]byte
			name    string
			values  []float32
			out, in int
		}{
			{&layers[l].selfQ, "attn.query", L.SelfQWeight, d, d}, {&layers[l].selfK, "attn.key", L.SelfKWeight, d, d},
			{&layers[l].selfV, "attn.value", L.SelfVWeight, d, d}, {&layers[l].selfO, "attn.out", L.SelfOWeight, d, d},
			{&layers[l].crossQ, "cross_attn.query", L.CrossQWeight, d, d}, {&layers[l].crossO, "cross_attn.out", L.CrossOWeight, d, d},
			{&layers[l].fc1, "mlp.0", L.FC1Weight, f, d}, {&layers[l].fc2, "mlp.2", L.FC2Weight, d, f},
		} {
			raw, err := load(p+m.name+".weight", m.values, m.out, m.in)
			if err != nil {
				return err
			}
			*m.dst = raw
		}
	}
	head, err := load("decoder.token_embedding.weight", dec.TokenEmbed, cfg.VocabSize, d)
	if err != nil {
		return err
	}
	dec.q5Layers, dec.lmHeadQ5 = layers, head
	return nil
}

// linearQ5Into is linearInto over packed Q5_0 rows: out[o] = SdotQ5_0 + bias[o].
// Rows are partitioned across workers; each output's reduction is unchanged.
func linearQ5Into(out, x []float32, raw []byte, bias []float32, inDim, outDim int) {
	rowBytes := inDim / 32 * simdrt.Q5_0BlockBytes
	_, _, _ = out[:outDim], x[:inDim], raw[:outDim*rowBytes]
	rows := func(lo, hi int) {
		for o := lo; o < hi; o++ {
			sum := simdrt.SdotQ5_0(x[:inDim], raw[o*rowBytes:(o+1)*rowBytes])
			if bias != nil && o < len(bias) {
				sum += bias[o]
			}
			out[o] = sum
		}
	}
	workers := max(1, min(min(linearWorkers, runtime.GOMAXPROCS(0)), outDim/256))
	if workers == 1 {
		rows(0, outDim)
		return
	}
	chunk := (outDim + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < outDim; lo += chunk {
		wg.Add(1)
		go func(lo, hi int) { defer wg.Done(); rows(lo, hi) }(lo, min(lo+chunk, outDim))
	}
	wg.Wait()
}

// decoderLinear uses the attached Q5 rows when present, else the F32 path.
func decoderLinear(out, x, weight []float32, raw []byte, bias []float32, inDim, outDim int) {
	if raw != nil {
		linearQ5Into(out, x, raw, bias, inDim, outDim)
		return
	}
	linearInto(out, x, weight, bias, inDim, outDim)
}
