package simd

import (
	"fmt"
	"math"
	"slices"
	"sync"
	"testing"
)

// Two explicit conversions through float64 make this independent of compiler
// contraction and the assembly implementation. Each operation rounds to F32.
func mulAddReference(dst, a, b float32) float32 {
	p := float32(float64(a) * float64(b))
	return float32(float64(dst) + float64(p))
}

func mulAddFixture(n int) (dst, a, b []float32) {
	dst, a, b = make([]float32, n), make([]float32, n), make([]float32, n)
	cases := [][3]uint32{
		{0xbf800000, 0x3f800001, 0x3f7ffffe}, // FMA differs: exact cancellation after rounded multiply
		{0xff7fffff, 0x7f7fffff, 0x40000000}, // product overflows before addition
		{0x80000000, 0x80000000, 0x3f800000}, // negative zero
		{0x00000000, 0x80000000, 0x3f800000},
		{0x00000001, 0x00000001, 0x3f000000}, // subnormal rounds to zero
		{0x00800000, 0x80000001, 0x3f800000},
		{0xff800000, 0x7f800000, 0x3f800000},
		{0x00000000, 0x7f800000, 0x00000000},
		{0x7fc12345, 0x3f800000, 0x40000000},
		{0x3f800000, 0x7fc12345, 0x40000000},
		{0x3f800000, 0x40000000, 0x7fc12345},
	}
	seed := uint32(991)
	for i := range dst {
		if i%2 == 0 {
			c := cases[(i/2)%len(cases)]
			dst[i], a[i], b[i] = math.Float32frombits(c[0]), math.Float32frombits(c[1]), math.Float32frombits(c[2])
		} else {
			for _, s := range [][]float32{dst, a, b} {
				seed = seed*1664525 + 1013904223
				s[i] = math.Float32frombits(seed)
			}
		}
	}
	return
}

func requireMulAddBits(t *testing.T, got, want []float32) {
	t.Helper()
	for i, v := range want {
		// NaN payload/sign selection is architecture-specific, not a numerical
		// guarantee. All other values, including signed zeros, are bit-exact.
		if math.IsNaN(float64(v)) && math.IsNaN(float64(got[i])) {
			continue
		}
		if math.Float32bits(got[i]) != math.Float32bits(v) {
			t.Fatalf("index %d: got %08x want %08x", i, math.Float32bits(got[i]), math.Float32bits(v))
		}
	}
}

func TestVecMulAddTo(t *testing.T) {
	old := HasVecAsm
	defer func() { HasVecAsm = old }()
	lengths := make([]int, 65, 66)
	for i := range lengths {
		lengths[i] = i + 1
	}
	lengths = append(lengths, 6144)
	for _, enabled := range []bool{false, old} {
		HasVecAsm = enabled
		for _, n := range lengths {
			for offset := 0; offset < 8; offset++ {
				t.Run(fmt.Sprintf("asm=%t/n=%d/offset=%d", enabled, n, offset), func(t *testing.T) {
					initial, aa, bb := mulAddFixture(n)
					// Exercise unaligned input pointers as well as the output.
					aStore, bStore := make([]float32, n+8), make([]float32, n+8)
					a, b := aStore[offset:offset+n], bStore[7-offset:7-offset+n]
					copy(a, aa)
					copy(b, bb)
					savedA, savedB := slices.Clone(a), slices.Clone(b)
					storage := make([]float32, n+16)
					for i := range storage {
						storage[i] = 19
					}
					dst := storage[offset+1 : offset+1+n]
					copy(dst, initial)
					want := make([]float32, n)
					for i := range want {
						want[i] = mulAddReference(initial[i], a[i], b[i])
					}
					if !VecMulAddTo(dst, a, b) {
						t.Fatal("valid call rejected")
					}
					requireMulAddBits(t, dst, want)
					for i := range a {
						if math.Float32bits(a[i]) != math.Float32bits(savedA[i]) || math.Float32bits(b[i]) != math.Float32bits(savedB[i]) {
							t.Fatal("read-only input changed")
						}
					}
					for i, v := range storage {
						if (i < offset+1 || i >= offset+1+n) && v != 19 {
							t.Fatal("guard overwritten")
						}
					}
				})
			}
		}
		for _, mode := range []string{"dst=a", "dst=b", "all", "inputs-overlap"} {
			for _, n := range []int{1, 7, 8, 9, 15, 16, 17, 65} {
				t.Run(fmt.Sprintf("asm=%t/%s/%d", enabled, mode, n), func(t *testing.T) {
					dst, a, b := mulAddFixture(n)
					switch mode {
					case "dst=a":
						a = dst
					case "dst=b":
						b = dst
					case "all":
						a, b = dst, dst
					case "inputs-overlap":
						_, data, _ := mulAddFixture(n + 1)
						a, b = data[:n], data[1:]
					}
					want := make([]float32, n)
					for i := range want {
						want[i] = mulAddReference(dst[i], a[i], b[i])
					}
					if !VecMulAddTo(dst, a, b) {
						t.Fatal("alias rejected")
					}
					requireMulAddBits(t, dst, want)
				})
			}
		}
		dst, a, b := mulAddFixture(65)
		if allocs := testing.AllocsPerRun(100, func() { VecMulAddTo(dst, a, b) }); allocs != 0 {
			t.Fatalf("allocations=%g", allocs)
		}
	}
}

func TestVecMulAddRejectsWithoutMutation(t *testing.T) {
	for _, sizes := range [][3]int{{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {1, 2, 1}, {1, 1, 2}, {2, 1, 2}, {2, 2, 1}} {
		dst, a, b := make([]float32, sizes[0]), make([]float32, sizes[1]), make([]float32, sizes[2])
		for i := range dst {
			dst[i] = 23
		}
		before := slices.Clone(dst)
		if VecMulAddTo(dst, a, b) {
			t.Fatalf("accepted lengths %v", sizes)
		}
		requireMulAddBits(t, dst, before)
	}
	for _, n := range []int{2, 7, 8, 9, 17} {
		for _, reverse := range []bool{false, true} {
			for _, input := range []int{0, 1} {
				store, a, b := mulAddFixture(n + 1)
				dst, overlap := store[:n], store[1:]
				if reverse {
					dst, overlap = overlap, dst
				}
				a, b = a[:n], b[:n]
				if input == 0 {
					a = overlap
				} else {
					b = overlap
				}
				before := slices.Clone(store)
				if VecMulAddTo(dst, a, b) {
					t.Fatal("accepted partial overlap")
				}
				requireMulAddBits(t, store, before)
			}
		}
	}
}

func TestVecMulAddSeparateRounding(t *testing.T) {
	a, b := math.Float32frombits(0x3f800001), math.Float32frombits(0x3f7ffffe)
	want := mulAddReference(-1, a, b)
	fused := float32(math.FMA(float64(a), float64(b), -1))
	if want == fused {
		t.Fatal("fixture must distinguish FMA")
	}
	// Repeat in every vector lane and the scalar tail.
	dst, aa, bb := make([]float32, 17), make([]float32, 17), make([]float32, 17)
	for i := range dst {
		dst[i], aa[i], bb[i] = -1, a, b
	}
	if !VecMulAddTo(dst, aa, bb) {
		t.Fatal("rejected")
	}
	for _, v := range dst {
		if math.Float32bits(v) != math.Float32bits(want) {
			t.Fatal("contracted FMA")
		}
	}
}

func TestVecMulAddConcurrent(t *testing.T) {
	initial, a, b := mulAddFixture(6145)
	want := make([]float32, len(a))
	for i := range want {
		want[i] = mulAddReference(initial[i], a[i], b[i])
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			dst := slices.Clone(initial)
			if !VecMulAddTo(dst, a, b) {
				t.Error("rejected")
			}
			requireMulAddBits(t, dst, want)
		}()
	}
	wg.Wait()
}

func BenchmarkVecMulAdd(b *testing.B) {
	for _, n := range []int{7, 8, 17, 128, 6144} {
		for _, path := range []string{"two-pass", "single-pass", "scalar"} {
			b.Run(fmt.Sprintf("%d/%s", n, path), func(b *testing.B) {
				dst, a, w, tmp := make([]float32, n), make([]float32, n), make([]float32, n), make([]float32, n)
				for i := range a {
					a[i], w[i] = .125, .0625
				}
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					switch path {
					case "two-pass":
						VecMul(tmp, a, w)
						VecAdd(dst, dst, tmp)
					case "single-pass":
						VecMulAddTo(dst, a, w)
					case "scalar":
						vecMulAddScalar(dst, a, w)
					}
				}
			})
		}
	}
}
