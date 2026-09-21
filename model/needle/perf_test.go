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

func BenchmarkNeedleAllocationMatrix(b *testing.B) {
	ctx := context.Background()
	cases := []struct {
		name, file string
		opts       Options
	}{
		{"v2-fp32", "needle2.json", Options{}},
		{"v2-cq-a8", "needle2-cq.json", Options{Quant: &Quantization{WeightBits: 4, ActivationBits: 8}}},
		{"v3-fp32", "needle3.json", Options{}},
		{"v3-cq-a8-kv8", "needle3-cq.json", Options{Quant: &Quantization{WeightBits: 4, ActivationBits: 8, KVBits: 8}}},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			m, f := fixtureFile(b, tc.file)
			b.Run("forward", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if _, err := m.Forward(f.Tokens, tc.opts); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("cached-prefix", func(b *testing.B) {
				d, err := m.NewDecoder(DecoderOptions{Capacity: len(f.Tokens), Execution: tc.opts})
				if err != nil {
					b.Fatal(err)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					d.Reset()
					for _, id := range f.Tokens {
						if _, err = d.Step(ctx, id); err != nil {
							b.Fatal(err)
						}
					}
				}
			})
			b.Run("loss-grad", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if _, _, err := m.LossGrad(f.Tokens, nil, tc.opts); err != nil {
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
					if _, _, err := m.AdapterLossGrad(a, f.Tokens, nil, tc.opts); err != nil {
						b.Fatal(err)
					}
				}
			})
		})
	}
	v2, v2f := loadV2Heads(b)
	for _, kind := range []HeadKind{Contrastive, Confidence} {
		b.Run("v2-head-"+string(kind), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := v2.Head(v2f.Tokens, kind, Options{}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
	v3, _, err := LoadArchive("../../loader/needle/testdata/needle3.cact")
	if err != nil {
		b.Fatal(err)
	}
	v3tokens := []int{2, 7, 0, 9, 3}
	for _, kind := range []HeadKind{Embedding, Confidence, Router} {
		b.Run("v3-head-"+string(kind), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := v3.Head(v3tokens, kind, Options{}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestInferenceArenaResetReusesBlocks(t *testing.T) {
	a := &inferenceArena{limit: 1 << 20}
	v := a.alloc(1024)
	ints := a.ints(100)
	for i := range v.x {
		v.x[i] = 1
	}
	for i := range ints {
		ints[i] = 7
	}
	floats, values, intBlocks, bytes := len(a.floatBlocks), len(a.valueBlocks), len(a.intBlocks), a.bytes
	allocs := testing.AllocsPerRun(100, func() {
		a.reset()
		x := a.alloc(1024)
		idx := a.ints(100)
		if x.x[0] != 0 || idx[0] != 0 {
			panic("stale arena data")
		}
	})
	if allocs != 0 {
		t.Fatalf("warm reset uses %.2f allocations", allocs)
	}
	if len(a.floatBlocks) != floats || len(a.valueBlocks) != values || len(a.intBlocks) != intBlocks || a.bytes != bytes {
		t.Fatal("arena grew during identical reuse")
	}
}
func TestTrainingValueUsesOneBackingArray(t *testing.T) {
	tp := &tape{train: true, limit: 1 << 20}
	allocs := testing.AllocsPerRun(100, func() {
		v := tp.alloc(2, 4)
		if len(v.x) != 8 || len(v.g) != 8 || cap(v.x) != 8 {
			panic("bad training value")
		}
	})
	// One value object and one combined x/g backing array; do not regress to three.
	if allocs != 2 {
		t.Fatalf("training value uses %.2f allocations, want 2", allocs)
	}
	v := tp.alloc(1, 2)
	v.x[0] = 1
	if v.g[0] != 0 {
		t.Fatal("gradient aliases value")
	}
}
