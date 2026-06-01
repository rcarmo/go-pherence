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
	if err := checkExpectedKVSmoke(3, 3, 2, 9440, 3, 3, 2, 9440); err != nil {
		t.Fatalf("expected KV smoke match: %v", err)
	}
	if err := checkExpectedKVSmoke(3, 3, 2, 9440, -1, -1, -1, -1); err != nil {
		t.Fatalf("unset expectations should be ignored: %v", err)
	}
	if err := checkExpectedKVSmoke(3, 3, 2, 9440, 4, -1, -1, -1); err == nil || !strings.Contains(err.Error(), "layer mismatch") {
		t.Fatalf("expected layer mismatch, got %v", err)
	}
	if err := checkExpectedKVSmoke(3, 3, 2, 9440, -1, 4, -1, -1); err == nil || !strings.Contains(err.Error(), "compressed count mismatch") {
		t.Fatalf("expected compressed mismatch, got %v", err)
	}
	if err := checkExpectedKVSmoke(3, 3, 2, 9440, -1, -1, 3, -1); err == nil || !strings.Contains(err.Error(), "full count mismatch") {
		t.Fatalf("expected full mismatch, got %v", err)
	}
	if err := checkExpectedKVSmoke(3, 3, 2, 9440, -1, -1, -1, 1); err == nil || !strings.Contains(err.Error(), "bytes mismatch") {
		t.Fatalf("expected bytes mismatch, got %v", err)
	}
}

func TestCheckExpectedRuntimeKV(t *testing.T) {
	plan := model.GGUFGenerationKVRuntimePlan{FloatKVBytesAllocated: 245760, EstimatedCompressedKVBytes: 81920}
	if err := checkExpectedRuntimeKV(plan, 245760, 81920); err != nil {
		t.Fatalf("expected runtime KV match: %v", err)
	}
	if err := checkExpectedRuntimeKV(plan, -1, -1); err != nil {
		t.Fatalf("unset expectations should be ignored: %v", err)
	}
	if err := checkExpectedRuntimeKV(plan, 1, -1); err == nil || !strings.Contains(err.Error(), "runtime F32 KV bytes mismatch") {
		t.Fatalf("expected runtime float mismatch, got %v", err)
	}
	if err := checkExpectedRuntimeKV(plan, -1, 1); err == nil || !strings.Contains(err.Error(), "runtime compressed KV bytes mismatch") {
		t.Fatalf("expected runtime compressed mismatch, got %v", err)
	}
}

func TestCheckExpectedBenchKV(t *testing.T) {
	stats := generationBenchStats{KVFloatBytes: 245760, KVCompressedBytes: 81920}
	if err := checkExpectedBenchKV(stats, 245760, 81920); err != nil {
		t.Fatalf("expected KV match: %v", err)
	}
	if err := checkExpectedBenchKV(stats, -1, -1); err != nil {
		t.Fatalf("unset expectations should be ignored: %v", err)
	}
	if err := checkExpectedBenchKV(stats, 1, -1); err == nil || !strings.Contains(err.Error(), "F32 KV bytes mismatch") {
		t.Fatalf("expected float KV mismatch, got %v", err)
	}
	if err := checkExpectedBenchKV(stats, -1, 1); err == nil || !strings.Contains(err.Error(), "compressed KV bytes mismatch") {
		t.Fatalf("expected compressed KV mismatch, got %v", err)
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
