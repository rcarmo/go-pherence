package needle

import (
	"encoding/json"
	"slices"
	"testing"

	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

func TestWidthSliceUpstream(t *testing.T) {
	f := headWidthFixture(t)
	m, e := New(&checkpoint.Checkpoint{FormatVersion: 2, Config: f.Width.Config, Tensors: f.Width.Tensors})
	if e != nil {
		t.Fatal(e)
	}
	child, e := m.SliceWidth(8)
	if e != nil {
		t.Fatal(e)
	}
	for key, w := range f.Width.ChildTensors {
		got, ok := child.tensors[key]
		if !ok || !slices.Equal(got.Shape, w.Shape) {
			t.Fatalf("shape mismatch %s", key)
		}
		compare(t, key, got.Data, w.Data, 0, 0)
	}
	for mode, w := range f.Width.Outputs {
		opts := Options{}
		if mode == "cq" {
			opts.Quant = &Quantization{WeightBits: 4, ActivationBits: 8, KVBits: 8}
		}
		out, e := child.Forward(f.Width.Tokens, opts)
		if e != nil {
			t.Fatal(e)
		}
		compare(t, "width logits", out, w.Logits.Data, 4e-5, 2e-3)
		for kind, want := range w.Heads {
			out, e := child.Head(f.Width.Tokens, kind, opts)
			if e != nil {
				t.Fatal(e)
			}
			compare(t, "width "+string(kind), out, want.Data, 1e-5, 3e-3)
		}
	}
	for key, w := range f.Width.Tensors {
		compare(t, "parent "+key, m.tensors[key].Data, w.Data, 0, 0)
	}
}
func TestWidthSliceAdmission(t *testing.T) {
	f := headWidthFixture(t)
	cp := &checkpoint.Checkpoint{FormatVersion: 2, Config: f.Width.Config, Tensors: f.Width.Tensors}
	m, e := New(cp)
	if e != nil {
		t.Fatal(e)
	}
	for _, w := range []int{0, 1, 3, 16, -8, int(^uint(0) >> 1)} {
		if _, e = m.SliceWidth(w); e == nil {
			t.Errorf("accepted width %d", w)
		}
	}
	var fields map[string]any
	if e = json.Unmarshal(cp.Config, &fields); e != nil {
		t.Fatal(e)
	}
	fields["ladder_widths"] = []int{}
	cp.Config, _ = json.Marshal(fields)
	m, e = New(cp)
	if e != nil {
		t.Fatal(e)
	}
	if _, e = m.SliceWidth(8); e == nil {
		t.Fatal("untrained width accepted")
	}
	v2, _ := fixtureFile(t, "needle2.json")
	if _, e = v2.SliceWidth(4); e == nil {
		t.Fatal("v2 width accepted")
	}
}

func TestWidthSourceRestrictions(t *testing.T) {
	f := headWidthFixture(t)
	cp := &checkpoint.Checkpoint{FormatVersion: 2, Config: f.Width.Config, Tensors: f.Width.Tensors}
	m, e := New(cp)
	if e != nil {
		t.Fatal(e)
	}
	owned := m.Checkpoint()
	owned.Tensors["ab_scales/embedding/embedding/a"] = checkpoint.Tensor{Shape: []int{1}, Data: []float32{1}}
	owned.Tensors["ab_scales/embedding/embedding/b"] = checkpoint.Tensor{Shape: []int{1}, Data: []float32{1}}
	m, e = New(owned)
	if e != nil {
		t.Fatal(e)
	}
	if _, e = m.SliceWidth(8); e == nil {
		t.Fatal("AB width accepted")
	}
	archive, _, e := LoadArchive("../../loader/needle/testdata/needle3.cact")
	if e != nil {
		t.Fatal(e)
	}
	if _, e = archive.SliceWidth(4); e == nil {
		t.Fatal("archive width accepted")
	}
}

func TestWidthRejectsUnlistedRung(t *testing.T) {
	f := headWidthFixture(t)
	var config map[string]any
	if err := json.Unmarshal(f.Width.Config, &config); err != nil {
		t.Fatal(err)
	}
	config["ladder_widths"] = []int{4}
	raw, _ := json.Marshal(config)
	m, err := New(&checkpoint.Checkpoint{FormatVersion: 2, Config: raw, Tensors: f.Width.Tensors})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = m.SliceWidth(8); err == nil {
		t.Fatal("unlisted width rung accepted")
	}
}
