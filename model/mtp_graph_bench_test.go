package model

import (
	"os"
	"path/filepath"
	"testing"
)

func BenchmarkGemma4MTPGraphCycleGGUF(b *testing.B) {
	root := findMTPGraphBenchRepoRoot()
	mainPath := filepath.Join(root, "models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf")
	drafterPath := filepath.Join(root, "models/gemma4-e4b-it-google-qat-gguf/MTP/gemma-4-E4B-it-BF16-MTP.gguf")
	if _, err := os.Stat(mainPath); err != nil {
		b.Skipf("local Gemma4 GGUF verifier not present: %v", err)
	}
	if _, err := os.Stat(drafterPath); err != nil {
		b.Skipf("local Gemma4 MTP drafter not present: %v", err)
	}
	m, err := LoadGemma4GGUFAsLlama(mainPath)
	if err != nil {
		b.Fatalf("load verifier: %v", err)
	}
	d, err := LoadGemma4MTPDrafterGGUF(drafterPath)
	if err != nil {
		b.Fatalf("load drafter: %v", err)
	}
	ctx, err := m.BuildMTPPromptContext([]int{10979})
	if err != nil {
		b.Fatalf("prompt context: %v", err)
	}
	ext, err := NewMTPDrafterExternalKVFromPromptContext(m, d, ctx)
	if err != nil {
		b.Fatalf("external KV: %v", err)
	}
	state, err := NewMTPDrafterState(236764, ctx.Activation, d.BackboneHiddenSize)
	if err != nil {
		b.Fatalf("drafter state: %v", err)
	}
	b.ReportAllocs()
	var outputTokens int
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decode, err := NewCPUDecodeStateFromMTPPromptContext(m, ctx, 3)
		if err != nil {
			b.Fatalf("decode state: %v", err)
		}
		step, err := decode.RunMTPGraphDecodeStep(d, state, ext, MTPGraphDecodeStepOptions{RemainingTokens: 3, DraftCount: 2}, MTPSpeculationStats{})
		if err != nil {
			b.Fatalf("graph step: %v", err)
		}
		if !sameInts(step.Commit.OutputTokens, []int{564, 236789, 236757}) {
			b.Fatalf("output tokens=%v", step.Commit.OutputTokens)
		}
		outputTokens += len(step.Commit.OutputTokens)
	}
	b.StopTimer()
	b.ReportMetric(float64(outputTokens)/b.Elapsed().Seconds(), "tok/s")
}

func findMTPGraphBenchRepoRoot() string {
	if wd, err := os.Getwd(); err == nil {
		for dir := wd; ; dir = filepath.Dir(dir) {
			if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
				return dir
			}
			if parent := filepath.Dir(dir); parent == dir {
				break
			}
		}
	}
	return "."
}
