//go:build amd64 && cgo

package llamaq4plan9_test

import (
	"math/rand"
	"testing"

	retained "github.com/rcarmo/go-pherence/loader/gguf/llamaq4"
	plan9 "github.com/rcarmo/go-pherence/loader/gguf/llamaq4plan9"
	"unsafe"
)

func TestCompilerPlan9TileMatchesRetained(t *testing.T) {
	if !retained.Available() {
		t.Skip("AVX2/AVX-VNNI/FMA unavailable")
	}
	rng := rand.New(rand.NewSource(20260809))
	for blocks := 1; blocks <= 80; blocks++ {
		q4 := make([]byte, blocks*144)
		q8 := make([]byte, blocks*136)
		_, _ = rng.Read(q4)
		_, _ = rng.Read(q8)
		for block := 0; block < blocks; block++ {
			for scale := 0; scale < 8; scale++ {
				q4[block*144+scale*2] = 0
				q4[block*144+scale*2+1] = 0x3c
			}
			for scale := 0; scale < 4; scale++ {
				q8[block*136+scale*2] = 0
				q8[block*136+scale*2+1] = 0x3c
			}
		}
		var want, got [32]float32
		if err := retained.DotQ4_0x8Q8_0x4VNNI(q4, q8, blocks, &want); err != nil {
			t.Fatal(err)
		}
		if err := plan9.DotQ4_0x8Q8_0x4CompilerPlan9(q4, q8, blocks, &got); err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("blocks=%d output=%d: want %08x got %08x", blocks, i, math32bits(want[i]), math32bits(got[i]))
			}
		}
	}
}

func BenchmarkTile(b *testing.B) {
	if !retained.Available() {
		b.Skip("AVX2/AVX-VNNI/FMA unavailable")
	}
	const blocks = 80
	q4 := make([]byte, blocks*144)
	q8 := make([]byte, blocks*136)
	for block := 0; block < blocks; block++ {
		for scale := 0; scale < 8; scale++ {
			q4[block*144+scale*2+1] = 0x3c
		}
		for scale := 0; scale < 4; scale++ {
			q8[block*136+scale*2+1] = 0x3c
		}
	}
	b.Run("retained-cgo", func(b *testing.B) {
		var out [32]float32
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := retained.DotQ4_0x8Q8_0x4VNNI(q4, q8, blocks, &out); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("compiler-plan9", func(b *testing.B) {
		var out [32]float32
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := plan9.DotQ4_0x8Q8_0x4CompilerPlan9(q4, q8, blocks, &out); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkProjection(b *testing.B) {
	if !retained.Available() {
		b.Skip("AVX2/AVX-VNNI/FMA unavailable")
	}
	const blocks, rows, tokens = 80, 128, 124
	q4 := make([]byte, (rows+7)/8*blocks*144)
	q8 := make([]byte, (tokens+3)/4*blocks*136)
	for group := 0; group < (rows+7)/8; group++ {
		for block := 0; block < blocks; block++ {
			for scale := 0; scale < 8; scale++ {
				off := (group*blocks+block)*144 + scale*2
				q4[off], q4[off+1] = 0, 0x3c
			}
		}
	}
	for panel := 0; panel < (tokens+3)/4; panel++ {
		for block := 0; block < blocks; block++ {
			for scale := 0; scale < 4; scale++ {
				off := (panel*blocks+block)*136 + scale*2
				q8[off], q8[off+1] = 0, 0x3c
			}
		}
	}
	out := make([]float32, rows*tokens)
	b.Run("retained-cgo", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if err := retained.ProjectQ4_0x8Q8_0x4VNNI(q4, q8, rows, tokens, blocks, out); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("staged-plan9", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if err := plan9.ProjectQ4_0x8Q8_0Stage(q4, q8, rows, tokens, blocks, out); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func math32bits(v float32) uint32 {
	return *(*uint32)(unsafe.Pointer(&v))
}

func TestCompilerPlan9DualPanelTileMatchesRetained(t *testing.T) {
	if !retained.Available() {
		t.Skip("AVX2/AVX-VNNI/FMA unavailable")
	}
	rng := rand.New(rand.NewSource(20260811))
	for blocks := 1; blocks <= 80; blocks++ {
		q4 := make([]byte, blocks*144)
		q8 := make([]byte, blocks*136*2)
		_, _ = rng.Read(q4)
		_, _ = rng.Read(q8)
		for block := 0; block < blocks; block++ {
			for scale := 0; scale < 8; scale++ {
				q4[block*144+scale*2] = 0
				q4[block*144+scale*2+1] = 0x3c
			}
			for panel := 0; panel < 2; panel++ {
				for scale := 0; scale < 4; scale++ {
					off := panel*blocks*136 + block*136 + scale*2
					q8[off] = 0
					q8[off+1] = 0x3c
				}
			}
		}
		want := make([]float32, 64)
		if err := retained.ProjectQ4_0x8Q8_0x4VNNI(q4, q8, 8, 8, blocks, want); err != nil {
			t.Fatal(err)
		}
		var got, staged [64]float32
		if err := plan9.DotQ4_0x8Q8_0x8CompilerPlan9(q4, q8, blocks, &got); err != nil {
			t.Fatal(err)
		}
		if err := plan9.DotQ4_0x8Q8_0x8StagePlan9(q4, q8, blocks, &staged); err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("blocks=%d output=%d: want %08x got %08x", blocks, i, math32bits(want[i]), math32bits(got[i]))
			}
			if want[i] != staged[i] {
				t.Fatalf("staged blocks=%d output=%d: want %08x got %08x", blocks, i, math32bits(want[i]), math32bits(staged[i]))
			}
		}
	}
}

func TestStageProjectionMatchesRetainedAcrossTails(t *testing.T) {
	if !retained.Available() {
		t.Skip("AVX2/AVX-VNNI/FMA unavailable")
	}
	rng := rand.New(rand.NewSource(20260812))
	for _, blocks := range []int{1, 3, 80} {
		for _, rows := range []int{1, 7, 8, 9, 15, 16} {
			for tokens := 1; tokens <= 17; tokens++ {
				q4 := make([]byte, (rows+7)/8*blocks*144)
				q8 := make([]byte, (tokens+3)/4*blocks*136)
				_, _ = rng.Read(q4)
				_, _ = rng.Read(q8)
				for group := 0; group < (rows+7)/8; group++ {
					for block := 0; block < blocks; block++ {
						for scale := 0; scale < 8; scale++ {
							off := (group*blocks+block)*144 + scale*2
							q4[off], q4[off+1] = 0, 0x3c
						}
					}
				}
				for panel := 0; panel < (tokens+3)/4; panel++ {
					for block := 0; block < blocks; block++ {
						for scale := 0; scale < 4; scale++ {
							off := (panel*blocks+block)*136 + scale*2
							q8[off], q8[off+1] = 0, 0x3c
						}
					}
				}
				want, got := make([]float32, rows*tokens), make([]float32, rows*tokens)
				if err := retained.ProjectQ4_0x8Q8_0x4VNNI(q4, q8, rows, tokens, blocks, want); err != nil {
					t.Fatal(err)
				}
				if err := plan9.ProjectQ4_0x8Q8_0Stage(q4, q8, rows, tokens, blocks, got); err != nil {
					t.Fatal(err)
				}
				for i := range want {
					if want[i] != got[i] {
						t.Fatalf("blocks=%d rows=%d tokens=%d output=%d: want %08x got %08x", blocks, rows, tokens, i, math32bits(want[i]), math32bits(got[i]))
					}
				}
			}
		}
	}
}

func BenchmarkFusedTile(b *testing.B) {
	if !retained.Available() {
		b.Skip("AVX2/AVX-VNNI/FMA unavailable")
	}
	const blocks = 80
	q4 := make([]byte, blocks*144)
	q8 := make([]byte, blocks*136*4)
	for block := 0; block < blocks; block++ {
		for scale := 0; scale < 8; scale++ {
			q4[block*144+scale*2+1] = 0x3c
		}
		for panel := 0; panel < 4; panel++ {
			for scale := 0; scale < 4; scale++ {
				q8[panel*blocks*136+block*136+scale*2+1] = 0x3c
			}
		}
	}
	b.Run("retained-cgo", func(b *testing.B) {
		out := make([]float32, 128)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := retained.ProjectQ4_0x8Q8_0x4VNNI(q4, q8, 8, 16, blocks, out); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("compiler-plan9-8x16", func(b *testing.B) {
		var out [128]float32
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := plan9.DotQ4_0x8Q8_0x16CompilerPlan9(q4, q8, blocks, &out); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("compiler-plan9-2x8x8", func(b *testing.B) {
		var out0, out1 [64]float32
		panelBytes := blocks * 136 * 2
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := plan9.DotQ4_0x8Q8_0x8CompilerPlan9(q4, q8[:panelBytes], blocks, &out0); err != nil {
				b.Fatal(err)
			}
			if err := plan9.DotQ4_0x8Q8_0x8CompilerPlan9(q4, q8[panelBytes:], blocks, &out1); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("staged-plan9-8x16-pair", func(b *testing.B) {
		var out [128]float32
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := plan9.DotQ4_0x8Q8_0x16PairPlan9(q4, q8, blocks, &out); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("staged-plan9-2x8x8", func(b *testing.B) {
		var out0, out1 [64]float32
		panelBytes := blocks * 136 * 2
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := plan9.DotQ4_0x8Q8_0x8StagePlan9(q4, q8[:panelBytes], blocks, &out0); err != nil {
				b.Fatal(err)
			}
			if err := plan9.DotQ4_0x8Q8_0x8StagePlan9(q4, q8[panelBytes:], blocks, &out1); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func TestCompilerPlan9FusedTileMatchesRetained(t *testing.T) {
	if !retained.Available() {
		t.Skip("AVX2/AVX-VNNI/FMA unavailable")
	}
	rng := rand.New(rand.NewSource(20260810))
	for blocks := 1; blocks <= 80; blocks++ {
		q4 := make([]byte, blocks*144)
		q8 := make([]byte, blocks*136*4)
		_, _ = rng.Read(q4)
		_, _ = rng.Read(q8)
		for block := 0; block < blocks; block++ {
			for scale := 0; scale < 8; scale++ {
				q4[block*144+scale*2] = 0
				q4[block*144+scale*2+1] = 0x3c
			}
			for panel := 0; panel < 4; panel++ {
				for scale := 0; scale < 4; scale++ {
					off := panel*blocks*136 + block*136 + scale*2
					q8[off] = 0
					q8[off+1] = 0x3c
				}
			}
		}
		want := make([]float32, 128)
		if err := retained.ProjectQ4_0x8Q8_0x4VNNI(q4, q8, 8, 16, blocks, want); err != nil {
			t.Fatal(err)
		}
		for name, run := range map[string]func(*[128]float32) error{
			"compiler": func(got *[128]float32) error { return plan9.DotQ4_0x8Q8_0x16CompilerPlan9(q4, q8, blocks, got) },
			"paired":   func(got *[128]float32) error { return plan9.DotQ4_0x8Q8_0x16PairPlan9(q4, q8, blocks, got) },
		} {
			var got [128]float32
			if err := run(&got); err != nil {
				t.Fatal(err)
			}
			for i := range want {
				if want[i] != got[i] {
					t.Fatalf("%s blocks=%d output=%d: want %08x got %08x", name, blocks, i, math32bits(want[i]), math32bits(got[i]))
				}
			}
		}
	}
}
