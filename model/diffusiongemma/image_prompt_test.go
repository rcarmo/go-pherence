package diffusiongemma

import "testing"

func TestExpandImagePlaceholderTokens(t *testing.T) {
	specials := SpecialTokenIDs{BOI: 10, IMAGE: 11, EOI: 12}
	got, err := ExpandImagePlaceholderTokens([]int{1, 11, 2}, specials, 3)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{1, 10, 11, 11, 11, 12, 2}
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d got=%v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%d want %d all=%v", i, got[i], want[i], got)
		}
	}
}

func TestExpandImagePlaceholderTokensTextOnlyCopies(t *testing.T) {
	in := []int{1, 2, 3}
	got, err := ExpandImagePlaceholderTokens(in, SpecialTokenIDs{BOI: 10, IMAGE: 11, EOI: 12}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(in) || &got[0] == &in[0] {
		t.Fatalf("expected copied text-only ids, got=%v in=%v", got, in)
	}
	for i := range in {
		if got[i] != in[i] {
			t.Fatalf("got[%d]=%d want %d", i, got[i], in[i])
		}
	}
}

func TestExpandImagePlaceholderTokensRejectsMissingConfig(t *testing.T) {
	if _, err := ExpandImagePlaceholderTokens([]int{11}, SpecialTokenIDs{BOI: -1, IMAGE: 11, EOI: 12}, 3); err == nil {
		t.Fatal("missing BOI accepted")
	}
	if _, err := ExpandImagePlaceholderTokens([]int{11}, SpecialTokenIDs{BOI: 10, IMAGE: 11, EOI: 12}, 0); err == nil {
		t.Fatal("zero soft tokens accepted")
	}
}

func TestLocalDiffusionGemmaImagePlaceholderExpansion(t *testing.T) {
	dir := localDiffusionGemmaModelDir(t)
	meta, err := LoadMetadata(dir)
	if err != nil {
		t.Fatal(err)
	}
	specials := meta.Tokenizer.SpecialTokenIDs(meta.Processor)
	got, err := ExpandImagePlaceholderTokens([]int{specials.BOT, specials.IMAGE, specials.EOT}, specials, meta.Shape.VisionSoftTokens)
	if err != nil {
		t.Fatal(err)
	}
	wantLen := 3 + meta.Shape.VisionSoftTokens + 1
	if len(got) != wantLen {
		t.Fatalf("len=%d want %d", len(got), wantLen)
	}
	if got[0] != specials.BOT || got[1] != specials.BOI || got[len(got)-1] != specials.EOT || got[len(got)-2] != specials.EOI {
		t.Fatalf("bad boundaries: first=%v last=%v", got[:3], got[len(got)-3:])
	}
	for i := 0; i < meta.Shape.VisionSoftTokens; i++ {
		if got[2+i] != specials.IMAGE {
			t.Fatalf("image slot %d=%d want %d", i, got[2+i], specials.IMAGE)
		}
	}
}
