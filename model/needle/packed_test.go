package needle

import (
	"context"
	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
	"testing"
)

func TestPackedArchiveParity(t *testing.T) {
	m, _, err := LoadArchive("../../loader/needle/testdata/needle3.cact")
	if err != nil {
		t.Fatal(err)
	}
	if m.PackedBytes() == 0 {
		t.Fatal("no packed weights")
	}
	ids := []int{2, 7, 4, 9, 3, 6, 5, 8, 7, 3, 5, 2}
	for _, layers := range []int{4, 2} {
		child := m
		if layers != 4 {
			child, err = m.SliceDepth(layers)
			if err != nil {
				t.Fatal(err)
			}
		}
		opts := Options{Packed: true}
		dec, err := child.NewDecoder(DecoderOptions{Capacity: len(ids), Execution: opts})
		if err != nil {
			t.Fatal(err)
		}
		for i, token := range ids {
			dense, err := child.Forward(ids[:i+1], Options{})
			if err != nil {
				t.Fatal(err)
			}
			packed, err := child.Forward(ids[:i+1], opts)
			if err != nil {
				t.Fatal(err)
			}
			compare(t, "packed dense", packed, dense, 1e-4, 5e-3)
			cached, err := dec.Step(context.Background(), token)
			if err != nil {
				t.Fatal(err)
			}
			compare(t, "packed cached", cached, dense[len(dense)-child.config.OutVocab:], 1e-4, 5e-3)
		}
		for _, kind := range []HeadKind{Embedding, Confidence, Router} {
			dense, err := child.Head(ids, kind, Options{})
			if err != nil {
				t.Fatal(err)
			}
			packed, err := child.Head(ids, kind, opts)
			if err != nil {
				t.Fatal(err)
			}
			compare(t, "packed head", packed, dense, 2e-5, 5e-3)
		}
	}
}
func TestPackedAdmission(t *testing.T) {
	m, f := fixture(t)
	if _, err := m.Forward(f.Tokens, Options{Packed: true}); err == nil {
		t.Fatal("source model accepted packed")
	}
	a, _, err := LoadArchive("../../loader/needle/testdata/needle3.cact")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = a.Forward(f.Tokens, Options{Packed: true, MaxWorkBytes: 1}); err == nil {
		t.Fatal("ignored workspace")
	}
	reloaded, err := New(a.Checkpoint())
	if err != nil {
		t.Fatal(err)
	}
	if _, err = reloaded.Forward(f.Tokens, Options{Packed: true}); err == nil {
		t.Fatal("decoded checkpoint falsely claimed packed")
	}
}
func BenchmarkArchivePackedDecode(b *testing.B) {
	m, _, err := LoadArchive("../../loader/needle/testdata/needle3.cact")
	if err != nil {
		b.Fatal(err)
	}
	for _, packed := range []bool{false, true} {
		name := "dense"
		if packed {
			name = "packed"
		}
		b.Run(name, func(b *testing.B) {
			d, err := m.NewDecoder(DecoderOptions{Capacity: 12, Execution: Options{Packed: packed}})
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			for b.Loop() {
				d.Reset()
				for _, token := range decoderFixtureTokens {
					if _, err = d.Step(context.Background(), token); err != nil {
						b.Fatal(err)
					}
				}
			}
		})
	}
}

func TestMappedArchiveOwnsCallerData(t *testing.T) {
	a, err := checkpoint.LoadArchive("../../loader/needle/testdata/needle3.cact")
	if err != nil {
		t.Fatal(err)
	}
	m, _, err := modelFromArchive(a)
	if err != nil {
		t.Fatal(err)
	}
	ids := []int{2, 7, 4}
	want, err := m.Forward(ids, Options{Packed: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range a.Records {
		clear(r.Data)
		clear(r.CQBlob)
		clear(r.Raw)
	}
	got, err := m.Forward(ids, Options{Packed: true})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "caller data ownership", got, want, 0, 0)
}
