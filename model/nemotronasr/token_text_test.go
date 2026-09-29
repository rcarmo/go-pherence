package nemotronasr

import (
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestDecodeRNNTTextReleasedVocabulary(t *testing.T) {
	vocab, err := tokenizer.Load(filepath.Join("..", "..", "checkpoints", "nemotron", "asr", "tokenizer.json"))
	if err != nil {
		t.Skipf("released tokenizer unavailable: %v", err)
	}
	for _, tc := range []struct {
		ids  []int
		want string
	}{
		{nil, ""},
		{[]int{rnntBlank, rnntBlank}, ""},
		{[]int{2860, 2, 1290}, "And so"},
		{[]int{2860, 2860, 2, 1290}, "And And so"},
		{[]int{2860, rnntBlank, 2860}, "And And"},
		{[]int{0, 1, 2860}, "And"},
	} {
		got, err := DecodeRNNTText(vocab, tc.ids)
		if err != nil || got != tc.want {
			t.Fatalf("ids=%v got=%q want=%q err=%v", tc.ids, got, tc.want, err)
		}
	}
	for _, ids := range [][]int{{-1}, {rnntBlank + 1}, {99999}} {
		if _, err := DecodeRNNTText(vocab, ids); err == nil {
			t.Fatalf("accepted invalid IDs %v", ids)
		}
	}
}
