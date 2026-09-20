package needle

import (
	"context"
	"os"
	"testing"
)

func BenchmarkNeedleTraining(b *testing.B) {
	m, f := fixture(b)
	b.Run("grad", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, _, err := m.LossGrad(f.Tokens, nil, Options{}); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("lora-grad", func(b *testing.B) {
		a, err := m.NewAdapter(2, 4, 1)
		if err != nil {
			b.Fatal(err)
		}
		b.ReportAllocs()
		for b.Loop() {
			if _, _, err := m.AdapterLossGrad(a, f.Tokens, nil, Options{}); err != nil {
				b.Fatal(err)
			}
		}
	})
}
func BenchmarkNeedleHeads(b *testing.B) {
	m, _, err := LoadArchive("../../loader/needle/testdata/needle3.cact")
	if err != nil {
		b.Fatal(err)
	}
	for _, kind := range []HeadKind{Embedding, Confidence, Router} {
		b.Run(string(kind), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := m.Head([]int{2, 7, 0, 9, 3}, kind, Options{}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// Opt-in released-model CPU profile; never downloads weights or touches a GPU.
func BenchmarkNeedleReleased(b *testing.B) {
	path := os.Getenv("GO_PHERENCE_NEEDLE_PROFILE_MODEL")
	if path == "" {
		b.Skip("set GO_PHERENCE_NEEDLE_PROFILE_MODEL to a local pinned archive")
	}
	m, tok, err := LoadArchive(path)
	if err != nil {
		b.Fatal(err)
	}
	ids, err := tok.Encode("<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n")
	if err != nil {
		b.Fatal(err)
	}
	_, _, bos, _ := tok.SpecialIDs()
	ids = append([]int{bos}, ids...)
	for _, packed := range []bool{false, true} {
		name := "dense"
		if packed {
			name = "packed"
		}
		b.Run(name, func(b *testing.B) {
			d, err := m.NewDecoder(DecoderOptions{Capacity: len(ids), Execution: Options{Packed: packed}})
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.SetBytes(int64(len(ids)))
			b.ResetTimer()
			for b.Loop() {
				d.Reset()
				for _, id := range ids {
					if _, err = d.Step(context.Background(), id); err != nil {
						b.Fatal(err)
					}
				}
			}
		})
	}
}

func TestArenaBudgetAndOwnedViews(t *testing.T) {
	tp := &tape{limit: 1 << 20, arena: &inferenceArena{limit: 1 << 20}}
	a := tp.alloc(1, 4)
	copy(a.x, []float32{1, 2, 3, 4})
	v := tp.slice(a, 1, 1, 2)
	for i := 0; i < 100; i++ {
		_ = tp.alloc(1, 100)
	}
	if v.x[0] != 2 || v.x[1] != 3 {
		t.Fatal("arena moved aliased view")
	}
	defer func() {
		if _, ok := recover().(workLimit); !ok {
			t.Error("arena budget not enforced")
		}
	}()
	tiny := &inferenceArena{limit: 100}
	tiny.alloc(100)
}
