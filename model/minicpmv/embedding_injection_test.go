package minicpmv

import "testing"

func TestInjectImageEmbeddings(t *testing.T) {
	plan := PromptPlan{NumQuery: 2, ImageSpans: []ImageSpan{{PatchStart: 1, PatchEnd: 3}}}
	tok := []float32{
		1, 1,
		2, 2,
		3, 3,
		4, 4,
	}
	img := []float32{20, 21, 30, 31}
	out, meta, err := InjectImageEmbeddings(tok, 4, 2, plan, img)
	if err != nil {
		t.Fatalf("InjectImageEmbeddings: %v", err)
	}
	want := []float32{1, 1, 20, 21, 30, 31, 4, 4}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("out[%d]=%v want %v; out=%v", i, out[i], want[i], out)
		}
	}
	if meta.Images != 1 || meta.ReplacedTokens != 2 || meta.HiddenSize != 2 || meta.SequenceLength != 4 {
		t.Fatalf("bad meta: %+v", meta)
	}
	if tok[2] != 2 {
		t.Fatalf("input embeddings were mutated: %v", tok)
	}
}

func TestInjectImageEmbeddingsMultipleImages(t *testing.T) {
	plan := PromptPlan{NumQuery: 1, ImageSpans: []ImageSpan{{PatchStart: 0, PatchEnd: 1}, {PatchStart: 2, PatchEnd: 3}}}
	tok := []float32{1, 2, 3}
	img := []float32{10, 30}
	out, meta, err := InjectImageEmbeddings(tok, 3, 1, plan, img)
	if err != nil {
		t.Fatalf("InjectImageEmbeddings: %v", err)
	}
	if got := []float32{out[0], out[1], out[2]}; got[0] != 10 || got[1] != 2 || got[2] != 30 || meta.ReplacedTokens != 2 {
		t.Fatalf("bad multi-image injection out=%v meta=%+v", got, meta)
	}
}

func TestInjectImageEmbeddingsRejectsShapeMismatch(t *testing.T) {
	plan := PromptPlan{NumQuery: 2, ImageSpans: []ImageSpan{{PatchStart: 1, PatchEnd: 3}}}
	if _, _, err := InjectImageEmbeddings(make([]float32, 8), 4, 2, plan, []float32{1, 2}); err == nil {
		t.Fatalf("expected image embedding shape mismatch")
	}
}

func TestInjectImageEmbeddingsRejectsBadSpan(t *testing.T) {
	plan := PromptPlan{NumQuery: 2, ImageSpans: []ImageSpan{{PatchStart: 2, PatchEnd: 5}}}
	if _, _, err := InjectImageEmbeddings(make([]float32, 8), 4, 2, plan, make([]float32, 4)); err == nil {
		t.Fatalf("expected bad span error")
	}
}
