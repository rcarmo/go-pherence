package needle

import (
	"encoding/json"
	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
	"os"
	"path/filepath"
	"testing"
)

func TestArchiveMappingAndRoundTrip(t *testing.T) {
	m, tok, err := LoadArchive("../../loader/needle/testdata/needle3.cact")
	if err != nil {
		t.Fatal(err)
	}
	if tok == nil || tok.VocabSize() != 280 {
		t.Fatal("tokenizer not loaded")
	}
	ids, err := tok.Encode("hello")
	if err != nil {
		t.Fatal(err)
	}
	out, err := m.Forward(ids, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(ids)*16 {
		t.Fatal("output vocab")
	}
	for _, kind := range []HeadKind{Embedding, Confidence, Router} {
		if _, err = m.Head(ids, kind, Options{}); err != nil {
			t.Fatal(err)
		}
	}
	if _, _, err = m.LossGrad([]int{2, 3}, nil, Options{}); err == nil {
		t.Fatal("archive training silently admitted")
	}
	if _, err = m.NewAdapter(2, 4, 1); err == nil {
		t.Fatal("archive LoRA silently admitted")
	}
	if _, err = m.Forward(ids, Options{Quant: &Quantization{WeightBits: 4}}); err == nil {
		t.Fatal("archive requantization")
	}
	path := filepath.Join(t.TempDir(), "decoded.safetensors")
	if err = checkpoint.Save(path, m.Checkpoint()); err != nil {
		t.Fatal(err)
	}
	reloaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	again, err := reloaded.Forward(ids, Options{})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "decoded archive roundtrip", again, out, 0, 0)
	child, err := m.SliceDepth(2)
	if err != nil {
		t.Fatal(err)
	}
	if !child.deployed {
		t.Fatal("slicing lost archive numerics")
	}
	if _, err = child.Forward(ids, Options{}); err != nil {
		t.Fatal(err)
	}
}

func TestArchiveMappedUpstream(t *testing.T) {
	b, err := os.ReadFile("../../loader/needle/testdata/archive-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Model struct {
			Tokens []int
			Logits checkpoint.Tensor
			Heads  map[HeadKind][]float32
		}
	}
	if err = json.Unmarshal(b, &ref); err != nil {
		t.Fatal(err)
	}
	m, _, err := LoadArchive("../../loader/needle/testdata/needle3.cact")
	if err != nil {
		t.Fatal(err)
	}
	out, err := m.Forward(ref.Model.Tokens, Options{})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "archive logits", out, ref.Model.Logits.Data, 4e-5, 3e-3)
	for kind, want := range ref.Model.Heads {
		out, err := m.Head(ref.Model.Tokens, kind, Options{})
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "archive "+string(kind), out, want, 1e-5, 3e-3)
	}
}
