package needle

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"io"
	"os"
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

func TestWidthEngramUpstream(t *testing.T) {
	file, err := os.Open("testdata/needle3-engram-width.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	gz, err := gzip.NewReader(file)
	if err != nil {
		t.Fatal(err)
	}
	defer gz.Close()
	var f struct {
		UpstreamPin    string                       `json:"upstream_pin"`
		Config         json.RawMessage              `json:"config"`
		Tensors        map[string]checkpoint.Tensor `json:"tensors"`
		Tokens         []int                        `json:"tokens"`
		ChildConfig    json.RawMessage              `json:"child_config"`
		ChildTensors   map[string]checkpoint.Tensor `json:"child_tensors"`
		ChildCQTensors map[string]checkpoint.Tensor `json:"child_cq_tensors"`
		Outputs        map[string]struct {
			Logits checkpoint.Tensor            `json:"logits"`
			Heads  map[string]checkpoint.Tensor `json:"heads"`
		} `json:"outputs"`
	}
	raw, err := io.ReadAll(io.LimitReader(gz, (64<<20)+1))
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) > 64<<20 {
		t.Fatal("fixture exceeds 64 MiB")
	}
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if f.UpstreamPin != "fc5bae0f9b6138828fe7589f6b531fb9a26968de" {
		t.Fatal("upstream pin changed")
	}
	tensors := map[string]checkpoint.Tensor{}
	for n, v := range f.Tensors {
		tensors[n] = checkpoint.Tensor{Shape: v.Shape, Data: v.Data}
	}
	m, err := New(&checkpoint.Checkpoint{FormatVersion: 2, Config: f.Config, Tensors: tensors})
	if err != nil {
		t.Fatal(err)
	}
	child, err := m.SliceWidth(512)
	if err != nil {
		t.Fatal(err)
	}
	if child.config.EngramHeads != 2 || child.config.EngramSeedHeads != 4 || len(child.config.EngramLayers) != 1 {
		t.Fatalf("child engram geometry %+v", child.config)
	}
	if len(child.tensors) != len(f.ChildTensors) {
		t.Fatal("child tensor count")
	}
	for n, w := range f.ChildTensors {
		got := child.tensors[n]
		if !slices.Equal(got.Shape, w.Shape) || !slices.Equal(got.Data, w.Data) {
			t.Fatalf("child tensor %s differs", n)
		}
	}
	for name, want := range f.ChildCQTensors {
		got := child.tensors[name]
		data := got.Data
		if isCQ(name) && len(got.Shape) >= 2 {
			data = make([]float32, len(got.Data))
			cqValues(data, got.Data, got.Shape, cqSecondLast(name), 4)
		}
		compare(t, "quantized child "+name, data, want.Data, 4e-5, 2e-3)
	}
	// The retained head seeds must remain parent-strided, not child-strided.
	for i, opts := range []Options{{}, {Quant: &Quantization{WeightBits: 4, ActivationBits: 8, KVBits: 8}}} {
		mode := "fp32"
		if i == 1 {
			mode = "cq"
		}
		want := f.Outputs[mode]
		logits, err := child.Forward(f.Tokens, opts)
		if err != nil {
			t.Fatal(err)
		}
		if mode == "cq" {
			golden, e := New(&checkpoint.Checkpoint{FormatVersion: 2, Config: f.ChildConfig, Tensors: f.ChildCQTensors})
			if e != nil {
				t.Fatal(e)
			}
			v, e := golden.Forward(f.Tokens, Options{Quant: &Quantization{ActivationBits: 8, KVBits: 8}})
			if e != nil {
				t.Fatal(e)
			}
			compare(t, "golden-dequant inference", v, want.Logits.Data, 4e-5, 2e-3)
		}
		compare(t, mode+" engram-width logits", logits, want.Logits.Data, 4e-5, 2e-3)
		for name, w := range want.Heads {
			got, err := child.Head(f.Tokens, HeadKind(name), opts)
			if err != nil {
				t.Fatal(err)
			}
			compare(t, mode+" engram-width "+name, got, w.Data, 1e-5, 3e-3)
		}
		d, err := child.NewDecoder(DecoderOptions{Capacity: len(f.Tokens), Execution: opts})
		if err != nil {
			t.Fatal(err)
		}
		for j, id := range f.Tokens {
			got, err := d.Step(context.Background(), id)
			if err != nil {
				t.Fatal(err)
			}
			all, err := child.Forward(f.Tokens[:j+1], opts)
			if err != nil {
				t.Fatal(err)
			}
			compare(t, mode+" cached engram-width", got, all[len(all)-child.config.OutVocab:], 4e-5, 2e-3)
		}
	}
	for n, w := range f.Tensors {
		if !slices.Equal(m.tensors[n].Data, w.Data) {
			t.Fatalf("parent changed: %s", n)
		}
	}
}
