package minicpmv

import "testing"

func TestInjectAudioEmbeddings(t *testing.T) {
	plan := AudioPromptPlan{PatchTokens: 2, AudioSpans: []AudioSpan{{PatchStart: 1, PatchEnd: 3}}}
	tok := []float32{1, 1, 2, 2, 3, 3, 4, 4}
	audio := []float32{20, 21, 30, 31}
	out, meta, err := InjectAudioEmbeddings(tok, 4, 2, plan, audio)
	if err != nil {
		t.Fatalf("InjectAudioEmbeddings: %v", err)
	}
	want := []float32{1, 1, 20, 21, 30, 31, 4, 4}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("out[%d]=%v want %v; out=%v", i, out[i], want[i], out)
		}
	}
	if meta.Audios != 1 || meta.ReplacedTokens != 2 || meta.HiddenSize != 2 || meta.SequenceLength != 4 {
		t.Fatalf("bad meta: %+v", meta)
	}
	if tok[2] != 2 {
		t.Fatalf("input embeddings were mutated: %v", tok)
	}
}

func TestInjectAudioEmbeddingsRejectsShapeMismatch(t *testing.T) {
	plan := AudioPromptPlan{PatchTokens: 2, AudioSpans: []AudioSpan{{PatchStart: 1, PatchEnd: 3}}}
	if _, _, err := InjectAudioEmbeddings(make([]float32, 8), 4, 2, plan, []float32{1, 2}); err == nil {
		t.Fatalf("expected audio embedding shape mismatch")
	}
}

func TestInjectAudioEmbeddingsRejectsBadSpan(t *testing.T) {
	plan := AudioPromptPlan{PatchTokens: 2, AudioSpans: []AudioSpan{{PatchStart: 2, PatchEnd: 5}}}
	if _, _, err := InjectAudioEmbeddings(make([]float32, 8), 4, 2, plan, make([]float32, 4)); err == nil {
		t.Fatalf("expected bad span error")
	}
}
