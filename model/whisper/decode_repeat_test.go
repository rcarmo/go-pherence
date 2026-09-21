package whisper

import "testing"

func TestSuppressTokenIDs(t *testing.T) {
	logits := []float32{1, 2, 3}
	suppressTokenIDs(logits, []int{1, 99, -1})
	if logits[1] > -1e20 || logits[0] != 1 || logits[2] != 3 {
		t.Fatalf("logits after suppress=%v", logits)
	}
}

func TestRepeatedNGram(t *testing.T) {
	if repeatedNGram([]int{1, 2, 1, 2}, 1, 3) {
		t.Fatal("treated bigram repeat as trigram repeat")
	}
	if !repeatedNGram([]int{1, 2, 3, 4, 1, 2}, 3, 3) {
		t.Fatal("missed repeated trigram")
	}
	if repeatedNGram([]int{1, 2, 3, 4, 1, 2}, 5, 3) {
		t.Fatal("false repeated trigram")
	}
}
