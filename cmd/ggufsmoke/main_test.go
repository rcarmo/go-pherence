package main

import (
	"strings"
	"testing"
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

func TestCheckExpectedDecodedRequiresTokenizer(t *testing.T) {
	if err := checkExpectedDecoded([]int{489}, nil, ""); err != nil {
		t.Fatalf("empty decoded expectation should be ignored: %v", err)
	}
	if err := checkExpectedDecoded([]int{489}, nil, "ype"); err == nil || !strings.Contains(err.Error(), "requires a GGUF tokenizer") {
		t.Fatalf("expected tokenizer error, got %v", err)
	}
}
