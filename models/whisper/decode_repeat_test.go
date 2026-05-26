package whisper

import "testing"

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
