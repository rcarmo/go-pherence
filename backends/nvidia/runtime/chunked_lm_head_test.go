package nvidia

import "testing"

func TestChunkedLMHeadRejectsMalformedInputs(t *testing.T) {
	if ChunkedLMHead(nil, nil, nil, 1, 1) {
		t.Fatal("accepted missing logits/hidden")
	}
	lmHead := []float32{1, 2, 3, 4}
	if ChunkedLMHead(make([]float32, 1), []float32{1, 2}, lmHead, 2, 2) {
		t.Fatal("accepted short logits")
	}
	if ChunkedLMHead(make([]float32, 2), []float32{1}, lmHead, 2, 2) {
		t.Fatal("accepted short hidden")
	}
	if ChunkedLMHead(make([]float32, 2), []float32{1, 2}, lmHead, 2, 0) {
		t.Fatal("accepted zero hidden size")
	}
	if ChunkedLMHead(make([]float32, 2), []float32{1, 2}, []float32{1}, 2, 2) {
		t.Fatal("accepted short LM head backing data")
	}
}
