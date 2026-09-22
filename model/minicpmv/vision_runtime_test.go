package minicpmv

import "testing"

func TestVisionEmbeddingCPUEncodeAndInject(t *testing.T) {
	cfg := tinySigLIPConfig()
	cfg.NumQuery = 1
	cfg.ResamplerConfig = tinyResamplerConfig().ResamplerConfig
	vision, err := LoadSigLIPVisionCPU(tinySigLIPSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	resampler, err := LoadResamplerCPU(tinyResamplerSource(cfg, false), cfg)
	if err != nil {
		t.Fatal(err)
	}
	runtime, err := NewVisionEmbeddingCPU(vision, resampler)
	if err != nil {
		t.Fatal(err)
	}
	pixels := []float32{1, 0, 0}
	tokens := []float32{
		10, 11, 12, 13,
		20, 21, 22, 23,
		30, 31, 32, 33,
	}
	plan := PromptPlan{NumQuery: 1, ImageSpans: []ImageSpan{{PatchStart: 1, PatchEnd: 2}}}
	out, meta, err := runtime.EncodeAndInject(pixels, [4]int{1, 3, 1, 1}, tokens, 3, 4, plan)
	if err != nil {
		t.Fatal(err)
	}
	if meta.ReplacedTokens != 1 || out[0] != 10 || out[8] != 30 || out[4] == 20 {
		t.Fatalf("output/meta=%v %+v", out, meta)
	}
	if tokens[4] != 20 {
		t.Fatal("input token embeddings mutated")
	}
}

func TestVisionEmbeddingCPURejectsInvalidComposition(t *testing.T) {
	cfg := tinySigLIPConfig()
	cfg.NumQuery = 1
	cfg.ResamplerConfig = tinyResamplerConfig().ResamplerConfig
	vision, err := LoadSigLIPVisionCPU(tinySigLIPSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	resampler, err := LoadResamplerCPU(tinyResamplerSource(cfg, false), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewVisionEmbeddingCPU(nil, resampler); err == nil {
		t.Fatal("accepted nil vision")
	}
	mismatch := *resampler
	mismatch.kvDim++
	if _, err := NewVisionEmbeddingCPU(vision, &mismatch); err == nil {
		t.Fatal("accepted hidden mismatch")
	}
	runtime, err := NewVisionEmbeddingCPU(vision, resampler)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := runtime.EncodeAndInject([]float32{1, 0, 0}, [4]int{1, 3, 1, 1}, make([]float32, 12), 3, 4, PromptPlan{NumQuery: 1}); err == nil {
		t.Fatal("accepted missing image span")
	}
}
