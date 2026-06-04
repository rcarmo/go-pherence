package main

import (
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/model"
)

func TestCheckExpectedGenerated(t *testing.T) {
	if err := checkExpectedGenerated([]int{489}, "489"); err != nil {
		t.Fatalf("expected match: %v", err)
	}
	if err := checkExpectedGenerated([]int{1, 2}, " 1, 2 "); err != nil {
		t.Fatalf("expected spaced match: %v", err)
	}
	if err := checkExpectedGenerated([]int{489}, ""); err != nil {
		t.Fatalf("empty expectation should be ignored: %v", err)
	}
	if err := checkExpectedGenerated([]int{489}, "488"); err == nil || !strings.Contains(err.Error(), "generated mismatch") {
		t.Fatalf("expected mismatch error, got %v", err)
	}
	if err := checkExpectedGenerated([]int{489}, "bad"); err == nil || !strings.Contains(err.Error(), "bad -expect-generated") {
		t.Fatalf("expected parse error, got %v", err)
	}
}

func TestCheckExpectedKVSmoke(t *testing.T) {
	if err := checkExpectedKVSmoke(3, 3, 2, 9440, 100, 9540, 3, 3, 2, 9440, 100, 9540); err != nil {
		t.Fatalf("expected KV smoke match: %v", err)
	}
	if err := checkExpectedKVSmoke(3, 3, 2, 9440, 100, 9540, -1, -1, -1, -1, -1, -1); err != nil {
		t.Fatalf("unset expectations should be ignored: %v", err)
	}
	if err := checkExpectedKVSmoke(3, 3, 2, 9440, 100, 9540, 4, -1, -1, -1, -1, -1); err == nil || !strings.Contains(err.Error(), "layer mismatch") {
		t.Fatalf("expected layer mismatch, got %v", err)
	}
	if err := checkExpectedKVSmoke(3, 3, 2, 9440, 100, 9540, -1, 4, -1, -1, -1, -1); err == nil || !strings.Contains(err.Error(), "compressed count mismatch") {
		t.Fatalf("expected compressed mismatch, got %v", err)
	}
	if err := checkExpectedKVSmoke(3, 3, 2, 9440, 100, 9540, -1, -1, 3, -1, -1, -1); err == nil || !strings.Contains(err.Error(), "full count mismatch") {
		t.Fatalf("expected full mismatch, got %v", err)
	}
	if err := checkExpectedKVSmoke(3, 3, 2, 9440, 100, 9540, -1, -1, -1, 1, -1, -1); err == nil || !strings.Contains(err.Error(), "stored bytes mismatch") {
		t.Fatalf("expected stored bytes mismatch, got %v", err)
	}
	if err := checkExpectedKVSmoke(3, 3, 2, 9440, 100, 9540, -1, -1, -1, -1, 1, -1); err == nil || !strings.Contains(err.Error(), "scratch bytes mismatch") {
		t.Fatalf("expected scratch bytes mismatch, got %v", err)
	}
	if err := checkExpectedKVSmoke(3, 3, 2, 9440, 100, 9540, -1, -1, -1, -1, -1, 1); err == nil || !strings.Contains(err.Error(), "total bytes mismatch") {
		t.Fatalf("expected total bytes mismatch, got %v", err)
	}
}

func TestCheckExpectedRuntimeKV(t *testing.T) {
	plan := model.GGUFGenerationKVRuntimePlan{FloatKVBytesAllocated: 245760, EstimatedCompressedKVBytes: 81920, EstimatedScratchBytes: 1280, EstimatedTotalBytes: 328960}
	if err := checkExpectedRuntimeKV(plan, 245760, 81920, 1280, 328960); err != nil {
		t.Fatalf("expected runtime KV match: %v", err)
	}
	if err := checkExpectedRuntimeKV(plan, -1, -1, -1, -1); err != nil {
		t.Fatalf("unset expectations should be ignored: %v", err)
	}
	if err := checkExpectedRuntimeKV(plan, 1, -1, -1, -1); err == nil || !strings.Contains(err.Error(), "runtime F32 KV bytes mismatch") {
		t.Fatalf("expected runtime float mismatch, got %v", err)
	}
	if err := checkExpectedRuntimeKV(plan, -1, 1, -1, -1); err == nil || !strings.Contains(err.Error(), "runtime compressed KV bytes mismatch") {
		t.Fatalf("expected runtime compressed mismatch, got %v", err)
	}
	if err := checkExpectedRuntimeKV(plan, -1, -1, 1, -1); err == nil || !strings.Contains(err.Error(), "runtime scratch bytes mismatch") {
		t.Fatalf("expected runtime scratch mismatch, got %v", err)
	}
	if err := checkExpectedRuntimeKV(plan, -1, -1, -1, 1); err == nil || !strings.Contains(err.Error(), "runtime total bytes mismatch") {
		t.Fatalf("expected runtime total mismatch, got %v", err)
	}
}

func TestCheckExpectedBenchKV(t *testing.T) {
	stats := generationBenchStats{KVCompressedLayers: 10, KVSeqLen: 2, KVCompressedCount: 0, KVFullCount: 20, KVFloatBytes: 245760, KVCompressedBytes: 81920, KVScratchBytes: 96768, KVTotalBytes: 424448}
	if err := checkExpectedBenchKV(stats, 10, 2, 0, 20, 245760, 81920, 96768, 424448); err != nil {
		t.Fatalf("expected KV match: %v", err)
	}
	if err := checkExpectedBenchKV(stats, -1, -1, -1, -1, -1, -1, -1, -1); err != nil {
		t.Fatalf("unset expectations should be ignored: %v", err)
	}
	if err := checkExpectedBenchKV(stats, 9, -1, -1, -1, -1, -1, -1, -1); err == nil || !strings.Contains(err.Error(), "compressed KV layers mismatch") {
		t.Fatalf("expected layers mismatch, got %v", err)
	}
	if err := checkExpectedBenchKV(stats, -1, 1, -1, -1, -1, -1, -1, -1); err == nil || !strings.Contains(err.Error(), "compressed KV seq mismatch") {
		t.Fatalf("expected seq mismatch, got %v", err)
	}
	if err := checkExpectedBenchKV(stats, -1, -1, 1, -1, -1, -1, -1, -1); err == nil || !strings.Contains(err.Error(), "compressed KV count mismatch") {
		t.Fatalf("expected compressed count mismatch, got %v", err)
	}
	if err := checkExpectedBenchKV(stats, -1, -1, -1, 1, -1, -1, -1, -1); err == nil || !strings.Contains(err.Error(), "full KV count mismatch") {
		t.Fatalf("expected full count mismatch, got %v", err)
	}
	if err := checkExpectedBenchKV(stats, -1, -1, -1, -1, 1, -1, -1, -1); err == nil || !strings.Contains(err.Error(), "F32 KV bytes mismatch") {
		t.Fatalf("expected float KV mismatch, got %v", err)
	}
	if err := checkExpectedBenchKV(stats, -1, -1, -1, -1, -1, 1, -1, -1); err == nil || !strings.Contains(err.Error(), "compressed KV bytes mismatch") {
		t.Fatalf("expected compressed KV mismatch, got %v", err)
	}
	if err := checkExpectedBenchKV(stats, -1, -1, -1, -1, -1, -1, 1, -1); err == nil || !strings.Contains(err.Error(), "scratch KV bytes mismatch") {
		t.Fatalf("expected scratch KV mismatch, got %v", err)
	}
	if err := checkExpectedBenchKV(stats, -1, -1, -1, -1, -1, -1, -1, 1); err == nil || !strings.Contains(err.Error(), "total KV bytes mismatch") {
		t.Fatalf("expected total KV mismatch, got %v", err)
	}
}

func TestCheckExpectedDecodedRequiresTokenizer(t *testing.T) {
	if err := checkExpectedDecoded([]int{489}, nil, ""); err != nil {
		t.Fatalf("empty decoded expectation should be ignored: %v", err)
	}
	if err := checkExpectedDecoded([]int{489}, nil, "ype"); err == nil || !strings.Contains(err.Error(), "requires a GGUF tokenizer") {
		t.Fatalf("expected tokenizer error, got %v", err)
	}
}
