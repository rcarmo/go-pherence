package needle

import (
	"context"
	"encoding/json"
	"os"
	"slices"
	"testing"

	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

func v2ArchiveConfig(t *testing.T) []byte {
	t.Helper()
	b, e := os.ReadFile("../../loader/needle/testdata/needle2-archive-config.json")
	if e != nil {
		t.Fatal(e)
	}
	return b
}
func TestNeedle2ArchiveUpstream(t *testing.T) {
	path := "../../loader/needle/testdata/needle2.cact"
	m, tok, err := LoadArchiveWithConfig(path, v2ArchiveConfig(t))
	if err != nil {
		t.Fatal(err)
	}
	if tok == nil || tok.VocabSize() != 280 || !m.deployed || m.config.Generation != 2 {
		t.Fatal("metadata")
	}
	var f struct {
		Tokens []int                          `json:"tokens"`
		Logits checkpoint.Tensor              `json:"logits"`
		Heads  map[HeadKind]checkpoint.Tensor `json:"heads"`
	}
	raw, err := os.ReadFile("../../loader/needle/testdata/needle2-archive-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	for _, packed := range []bool{false, true} {
		opts := Options{Packed: packed}
		got, err := m.Forward(f.Tokens, opts)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "archive logits", got, f.Logits.Data, 3e-5, 1e-3)
		for kind, w := range f.Heads {
			out, err := m.Head(f.Tokens, kind, opts)
			if err != nil {
				t.Fatal(err)
			}
			compare(t, string(kind), out, w.Data, 1e-5, 3e-3)
		}
		d, err := m.NewDecoder(DecoderOptions{Execution: opts})
		if err != nil {
			t.Fatal(err)
		}
		for i, id := range f.Tokens {
			out, err := d.Step(context.Background(), id)
			if err != nil {
				t.Fatal(err)
			}
			want, err := m.Forward(f.Tokens[:i+1], opts)
			if err != nil {
				t.Fatal(err)
			}
			compare(t, "cache", out, want[len(want)-m.config.OutVocab:], 3e-5, 1e-3)
		}
		a, err := m.Generate(context.Background(), f.Tokens, 8, -1, opts)
		if err != nil {
			t.Fatal(err)
		}
		b, err := m.GenerateCached(context.Background(), f.Tokens, 8, -1, DecoderOptions{Execution: opts})
		if err != nil || !slices.Equal(a, b) {
			t.Fatalf("generation %v != %v %v", a, b, err)
		}
	}
	if _, ok := m.tensors["contrastive_head/log_temp"]; ok {
		t.Fatal("invented missing temperature")
	}
	if _, err = m.NewAdapter(2, 4, 1); err == nil {
		t.Fatal("archive LoRA")
	}
	if _, _, err = m.LossGrad(f.Tokens, nil, Options{}); err == nil {
		t.Fatal("archive training")
	}
	cp := m.Checkpoint()
	reloaded, err := New(cp)
	if err != nil {
		t.Fatal(err)
	}
	want, err := m.Forward(f.Tokens, Options{})
	if err != nil {
		t.Fatal(err)
	}
	got, err := reloaded.Forward(f.Tokens, Options{})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "decoded reload", got, want, 0, 0)
	for kind, w := range f.Heads {
		out, err := reloaded.Head(f.Tokens, kind, Options{})
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "reload head", out, w.Data, 1e-5, 3e-3)
	}
}
func TestNeedle2ArchiveAdmission(t *testing.T) {
	path := "../../loader/needle/testdata/needle2.cact"
	config := v2ArchiveConfig(t)
	if _, _, err := LoadArchive(path); err == nil {
		t.Fatal("missing sidecar")
	}
	if _, _, err := LoadArchiveWithConfig("../../loader/needle/testdata/needle3.cact", config); err == nil {
		t.Fatal("v3 sidecar")
	}
	var fields map[string]any
	if err := json.Unmarshal(config, &fields); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"num_heads", "rope_theta", "engram_layers", "contrastive_dim"} {
		copy := map[string]any{}
		for k, v := range fields {
			copy[k] = v
		}
		delete(copy, key)
		b, _ := json.Marshal(copy)
		if _, _, err := LoadArchiveWithConfig(path, b); err == nil {
			t.Fatal("missing " + key)
		}
	}
	for key, value := range map[string]any{"num_heads": 4, "generation": 3, "unknown": 1, "contrastive_dim": 8, "pad_token_id": 1} {
		copy := map[string]any{}
		for k, v := range fields {
			copy[k] = v
		}
		copy[key] = value
		b, _ := json.Marshal(copy)
		if _, _, err := LoadArchiveWithConfig(path, b); err == nil {
			t.Fatal("mismatch " + key)
		}
	}
	if _, err := parseArchiveConfigV2(append(config, []byte(`{}`)...)); err == nil {
		t.Fatal("trailing data")
	}
	a, err := checkpoint.LoadArchive(path)
	if err != nil {
		t.Fatal(err)
	}
	a.Header[4] = 4
	if _, _, err = mapArchiveConfig(a, false, config); err == nil {
		t.Fatal("CQ KV admitted")
	}
}

func TestNeedle2ArchiveOwnershipAndShapes(t *testing.T) {
	path := "../../loader/needle/testdata/needle2.cact"
	config := v2ArchiveConfig(t)
	a, err := checkpoint.LoadArchive(path)
	if err != nil {
		t.Fatal(err)
	}
	m, _, err := mapArchiveConfig(a, false, config)
	if err != nil {
		t.Fatal(err)
	}
	ids := []int{2, 7, 4}
	before, err := m.Forward(ids, Options{})
	if err != nil {
		t.Fatal(err)
	}
	for i := range a.Records {
		clear(a.Records[i].Data)
		clear(a.Records[i].CQBlob)
		clear(a.Records[i].Raw)
	}
	clear(config)
	after, err := m.Forward(ids, Options{})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "ownership", after, before, 0, 0)
	for _, mutation := range []string{"record", "headcode", "headbias", "trailing"} {
		t.Run(mutation, func(t *testing.T) {
			a, err := checkpoint.LoadArchive(path)
			if err != nil {
				t.Fatal(err)
			}
			switch mutation {
			case "record":
				a.Records[0].Shape = []int{280, 9}
			case "headcode":
				a.Records[43].Data = []float32{2, 1}
			case "headbias":
				a.Records[46].Data[0] = 1
			case "trailing":
				a.Records = append(a.Records, a.Records[0])
			}
			if _, _, err = mapArchiveConfig(a, false, v2ArchiveConfig(t)); err == nil {
				t.Fatal("malformed mapping admitted")
			}
		})
	}
}
