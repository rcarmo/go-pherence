package diffusiongemma

import (
	"strings"
	"testing"
)

func TestPromptAppendSuffix(t *testing.T) {
	op, ok := promptAppendOpportunity([]int{1, 2}, []int{1, 2, 3, 4})
	if !ok || op.PrefixTokens != 2 || op.SuffixTokens != 2 || op.NewTokens != 4 {
		t.Fatalf("bad opportunity=%+v ok=%v", op, ok)
	}
	suffix, ok := promptAppendSuffix([]int{1, 2}, []int{1, 2, 3, 4})
	if !ok || len(suffix) != 2 || suffix[0] != 3 || suffix[1] != 4 {
		t.Fatalf("bad append suffix=%v ok=%v", suffix, ok)
	}
	if _, ok := promptAppendSuffix([]int{1, 2}, []int{1, 9, 3}); ok {
		t.Fatalf("accepted non-prefix append")
	}
	if _, ok := promptAppendSuffix([]int{1, 2}, []int{1, 2}); ok {
		t.Fatalf("accepted unchanged prompt as append")
	}
}

func TestDenoiseGPUPromptPrefillPathIsUsed(t *testing.T) {
	d := &TextDenoiser{Shape: Shape{VocabSize: 8}, Dispatcher: GPUDispatcher{Progress: true}}
	_, err := d.Denoise(ForwardInput{PromptIDs: []int{10}, Canvas: []int{1}})
	if err == nil || !strings.Contains(err.Error(), "DiffusionGemma encoder missing weights") {
		t.Fatalf("expected GPU prompt prefill path to reach encoder weight validation, got %v", err)
	}
}

func TestDenoisePromptPrefillRequiresGPUBackend(t *testing.T) {
	d := &TextDenoiser{
		Shape:            Shape{VocabSize: 8},
		Dispatcher:       NotImplementedDispatcher{},
		EncoderKV:        []EncoderKVLayer{{SeqLen: 2}},
		EncoderPromptIDs: []int{10, 11},
		EncoderPromptLen: 2,
	}
	_, err := d.Denoise(ForwardInput{PromptIDs: []int{10, 11, 12}, Canvas: []int{1}})
	if err == nil || !strings.Contains(err.Error(), "prompt prefill requires a GPU backend implementation") {
		t.Fatalf("expected GPU backend prefill error, got %v", err)
	}
}

func TestIncrementalPromptKVEnabledByDefaultCanBeDisabled(t *testing.T) {
	if !diffusionGemmaIncrementalPromptKVEnabled() {
		t.Fatalf("incremental KV should default to enabled for GGUF-capable paths")
	}
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_DISABLE_INCREMENTAL_KV", "1")
	if diffusionGemmaIncrementalPromptKVEnabled() {
		t.Fatalf("incremental KV not disabled by env")
	}
}

func TestDenoiseRequireIncrementalPromptKVStillRequiresGPUPrefill(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_REQUIRE_INCREMENTAL_KV", "1")
	d := &TextDenoiser{
		Shape:            Shape{VocabSize: 8},
		Dispatcher:       NotImplementedDispatcher{},
		EncoderKV:        []EncoderKVLayer{{SeqLen: 2}},
		EncoderPromptIDs: []int{10, 11},
		EncoderPromptLen: 2,
	}
	_, err := d.Denoise(ForwardInput{PromptIDs: []int{10, 11, 12, 13}, Canvas: []int{1}})
	if err == nil || !strings.Contains(err.Error(), "prompt prefill requires a GPU backend implementation") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestDenoiseCPUIncrementalKVFallbackDisabled(t *testing.T) {
	d := &TextDenoiser{
		Shape:            Shape{VocabSize: 8},
		Dispatcher:       CPUDispatcher{GGUFExpertIndex: &GGUFExpertIndex{}},
		EncoderKV:        []EncoderKVLayer{{SeqLen: 2}},
		EncoderPromptIDs: []int{10, 11},
		EncoderPromptLen: 2,
	}
	_, err := d.Denoise(ForwardInput{PromptIDs: []int{10, 11, 12}, Canvas: []int{1}})
	if err == nil || !strings.Contains(err.Error(), "CPU prompt encoding is disabled") {
		t.Fatalf("expected CPU prompt encoding disabled error, got %v", err)
	}
}
