package qwen

import (
	"github.com/rcarmo/go-pherence/runtime/kv"
	"testing"
)

func TestPromptKeyRejectsInvalidChunkAndBoundsHugeChunk(t *testing.T) {
	tokens := []int{1, 2, 3}
	for _, chunk := range []int{0, -1} {
		if got := PromptPrefixKey("m", "l", "f32", tokens, chunk); got != (kv.ChunkKey{}) {
			t.Fatal(got)
		}
		if NewPromptCache(1024).Store("m", "l", "f32", tokens, chunk, PromptSnapshot{}) {
			t.Fatal("invalid chunk stored")
		}
	}
	maxInt := int(^uint(0) >> 1)
	got := PromptPrefixKey("m", "l", "f32", tokens, maxInt)
	if got.TokenHash != kv.HashTokenChunk(0, tokens) || got.EndPos != 3 || got.ChunkSize != maxInt {
		t.Fatal(got)
	}
	normal := PromptPrefixKey("m", "l", "f32", tokens, 2)
	want := kv.HashTokenChunk(kv.HashTokenChunk(0, tokens[:2]), tokens[2:])
	if normal.TokenHash != want {
		t.Fatal("valid key changed")
	}
}
