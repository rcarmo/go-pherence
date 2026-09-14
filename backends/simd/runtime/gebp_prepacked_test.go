package simd

import (
	"fmt"
	"math"
	"testing"
)

func TestPackSgemmNTWeightsLayout(t *testing.T) {
	bits := []uint32{0, 0x80000000, 0x3f800000, 0xbf800000, 0x7f7fffff, 1, 0x00800000, 0x4a123456}
	for _, shape := range []struct {
		n   int
		k   int
		pad int
	}{
		{n: 7, k: 5, pad: 3},
		{n: 16, k: 1, pad: 0},
		{n: 17, k: 7, pad: 2},
		{n: 33, k: 17, pad: 5},
	} {
		t.Run(fmt.Sprintf("n%d/k%d/pad%d", shape.n, shape.k, shape.pad), func(t *testing.T) {
			ldb := shape.k + shape.pad
			weightLen := (shape.n-1)*ldb + shape.k
			weights := make([]float32, weightLen)
			for i := range weights {
				weights[i] = math.Float32frombits(bits[i%len(bits)] ^ uint32(i%7))
			}
			fullPanels, packedLen, ok := checkedSgemmNTFullPanelLayout(shape.n, shape.k)
			if !ok {
				t.Fatal("unexpected packed layout overflow")
			}
			packed, err := PackSgemmNTWeights(weights, shape.n, shape.k, ldb)
			if err != nil {
				t.Fatalf("PackSgemmNTWeights error: %v", err)
			}
			if len(packed) != packedLen {
				t.Fatalf("packed len=%d want %d", len(packed), packedLen)
			}
			guarded := make([]float32, packedLen+2)
			for i := range guarded {
				guarded[i] = 12345
			}
			into, err := PackSgemmNTWeightsInto(weights, shape.n, shape.k, ldb, guarded[1:len(guarded)-1])
			if err != nil {
				t.Fatalf("PackSgemmNTWeightsInto error: %v", err)
			}
			if len(into) != packedLen {
				t.Fatalf("into len=%d want %d", len(into), packedLen)
			}
			if guarded[0] != 12345 || guarded[len(guarded)-1] != 12345 {
				t.Fatal("PackSgemmNTWeightsInto overwrote guard")
			}
			for i := range packed {
				if math.Float32bits(into[i]) != math.Float32bits(packed[i]) {
					t.Fatalf("into[%d]=%08x want %08x", i, math.Float32bits(into[i]), math.Float32bits(packed[i]))
				}
			}
			for panel := 0; panel < fullPanels; panel++ {
				expected := make([]float32, shape.k*gebpNR)
				packBNT(weights, ldb, panel*gebpNR, gebpNR, shape.k, expected)
				base := panel * shape.k * gebpNR
				for i := range expected {
					if math.Float32bits(packed[base+i]) != math.Float32bits(expected[i]) {
						t.Fatalf("panel %d elem %d=%08x want %08x", panel, i, math.Float32bits(packed[base+i]), math.Float32bits(expected[i]))
					}
				}
			}
		})
	}
}

func TestPackSgemmNTWeightsRejectsMalformed(t *testing.T) {
	if _, err := PackSgemmNTWeights(nil, 0, 1, 1); err == nil {
		t.Fatal("accepted zero n")
	}
	if _, err := PackSgemmNTWeights(make([]float32, 16), 16, 4, 3); err == nil {
		t.Fatal("accepted ldb < k")
	}
	if _, err := PackSgemmNTWeightsInto(make([]float32, 63), 16, 4, 4, make([]float32, 64)); err == nil {
		t.Fatal("accepted short weights")
	}
	if _, err := PackSgemmNTWeightsInto(make([]float32, 64), 16, 4, 4, make([]float32, 63)); err == nil {
		t.Fatal("accepted short packed buffer")
	}
	backing := make([]float32, 128)
	weights := backing[:64]
	overlap := backing[32:96]
	if _, err := PackSgemmNTWeightsInto(weights, 16, 4, 4, overlap); err == nil {
		t.Fatal("accepted overlapping packed/source buffers")
	}
	maxInt := int(^uint(0) >> 1)
	if _, err := PackSgemmNTWeights(nil, maxInt, 17, 17); err == nil {
		t.Fatal("accepted overflowing footprint")
	}
}

func TestSgemmNTPrepackedParity(t *testing.T) {
	for _, tc := range []struct {
		name   string
		m      int
		n      int
		k      int
		ldaPad int
		ldbPad int
		ldcPad int
		alpha  float32
	}{
		{name: "n-tail-only", m: gebpMR * 2, n: 33, k: 7, ldaPad: 0, ldbPad: 3, ldcPad: 5, alpha: 0.75},
		{name: "m-tail-only", m: gebpMR + 1, n: 32, k: 15, ldaPad: 2, ldbPad: 0, ldcPad: 3, alpha: -0.5},
		{name: "both-tails", m: gebpMR + 2, n: 17, k: 9, ldaPad: 3, ldbPad: 4, ldcPad: 7, alpha: 1.25},
		{name: "full-panels", m: gebpMR * 3, n: 48, k: 16, ldaPad: 1, ldbPad: 2, ldcPad: 4, alpha: 1},
		{name: "no-full-panel", m: gebpMR + 3, n: 15, k: 5, ldaPad: 2, ldbPad: 1, ldcPad: 2, alpha: -1},
		{name: "fallback-micro-m", m: gebpMR - 1, n: 16, k: 11, ldaPad: 4, ldbPad: 2, ldcPad: 3, alpha: 0.125},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lda := tc.k + tc.ldaPad
			ldb := tc.k + tc.ldbPad
			ldc := tc.n + tc.ldcPad
			a := make([]float32, (tc.m-1)*lda+tc.k)
			weights := make([]float32, (tc.n-1)*ldb+tc.k)
			baseC := make([]float32, (tc.m-1)*ldc+tc.n)
			for i := range a {
				a[i] = float32(i%19-9) * 0.125
			}
			for i := range weights {
				weights[i] = float32(i%23-11) * 0.0625
			}
			for i := range baseC {
				baseC[i] = float32(i%13-6) * 0.25
			}
			packed, err := PackSgemmNTWeights(weights, tc.n, tc.k, ldb)
			if err != nil {
				t.Fatalf("PackSgemmNTWeights error: %v", err)
			}
			want := append([]float32(nil), baseC...)
			got := append([]float32(nil), baseC...)
			if !SgemmNTPackedTo(want, a, weights, make([]float32, tc.k*gebpNR), tc.m, tc.n, tc.k, tc.alpha, lda, ldb, ldc) {
				t.Fatal("SgemmNTPackedTo rejected valid inputs")
			}
			if !SgemmNTPrepackedTo(got, a, weights, packed, tc.m, tc.n, tc.k, tc.alpha, lda, ldb, ldc) {
				t.Fatal("SgemmNTPrepackedTo rejected valid inputs")
			}
			for i := range want {
				if want[i] != got[i] {
					t.Fatalf("c[%d]=%08x want %08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
				}
			}
		})
	}
}

func TestSgemmNTPrepackedRejectsMalformed(t *testing.T) {
	const (
		m = 1
		n = 16
		k = 4
	)
	lda, ldb, ldc := 6, 5, 18
	a := make([]float32, (m-1)*lda+k)
	weights := make([]float32, (n-1)*ldb+k)
	c := make([]float32, (m-1)*ldc+n)
	if SgemmNTPrepackedTo(c, a, weights, nil, m, n, k, 1, lda, ldb, ldc) {
		t.Fatal("accepted missing packed buffer on fallback path")
	}
	if SgemmNTPrepackedTo(c, a[:len(a)-1], weights, make([]float32, k*gebpNR), m, n, k, 1, lda, ldb, ldc) {
		t.Fatal("accepted short A slice")
	}
	if SgemmNTPrepackedTo(c, a, weights[:len(weights)-1], make([]float32, k*gebpNR), m, n, k, 1, lda, ldb, ldc) {
		t.Fatal("accepted short weights slice")
	}
	if SgemmNTPrepackedTo(c[:len(c)-1], a, weights, make([]float32, k*gebpNR), m, n, k, 1, lda, ldb, ldc) {
		t.Fatal("accepted short C slice")
	}
	const mFull = gebpMR
	aFull := make([]float32, (mFull-1)*k+k)
	weightsFull := make([]float32, (n-1)*k+k)
	cFull := make([]float32, (mFull-1)*n+n)
	if SgemmNTPrepackedTo(cFull, aFull, weightsFull, make([]float32, k*gebpNR-1), mFull, n, k, 1, k, k, n) {
		t.Fatal("accepted short packed buffer")
	}
	if !SgemmNTPrepackedTo(c[:15], a, weights[:(15-1)*ldb+k], nil, 1, 15, k, 1, lda, ldb, 15) {
		t.Fatal("rejected valid no-full-panel fallback")
	}
}

func TestSgemmNTPrepackedNoAllocs(t *testing.T) {
	const (
		m = 126
		n = 1024
		k = 128
	)
	lda, ldb, ldc := k+3, k+5, n+7
	a := make([]float32, (m-1)*lda+k)
	weights := make([]float32, (n-1)*ldb+k)
	c := make([]float32, (m-1)*ldc+n)
	_, packedLen, ok := checkedSgemmNTFullPanelLayout(n, k)
	if !ok {
		t.Fatal("unexpected packed length overflow")
	}
	packed := make([]float32, packedLen)
	for i := range a {
		a[i] = float32(i%29-14) * 0.125
	}
	for i := range weights {
		weights[i] = float32(i%31-15) * 0.0625
	}
	if allocs := testing.AllocsPerRun(10, func() {
		if _, err := PackSgemmNTWeightsInto(weights, n, k, ldb, packed); err != nil {
			panic(err)
		}
	}); allocs != 0 {
		t.Fatalf("PackSgemmNTWeightsInto allocs %g", allocs)
	}
	if _, err := PackSgemmNTWeightsInto(weights, n, k, ldb, packed); err != nil {
		t.Fatalf("PackSgemmNTWeightsInto error: %v", err)
	}
	if allocs := testing.AllocsPerRun(10, func() {
		clear(c)
		if !SgemmNTPrepackedTo(c, a, weights, packed, m, n, k, 1, lda, ldb, ldc) {
			panic("SgemmNTPrepackedTo returned false")
		}
	}); allocs != 0 {
		t.Fatalf("SgemmNTPrepackedTo allocs %g", allocs)
	}
}

func BenchmarkSgemmNTPrepackedVsPackedOld(b *testing.B) {
	for _, m := range []int{126, 128} {
		const (
			n = 1024
			k = 1024
		)
		b.Run(fmt.Sprintf("m%d/n%d/k%d", m, n, k), func(b *testing.B) {
			a := make([]float32, m*k)
			weights := make([]float32, n*k)
			c := make([]float32, m*n)
			for i := range a {
				a[i] = float32(i%31-15) * 0.03125
			}
			for i := range weights {
				weights[i] = float32(i%29-14) * 0.03125
			}
			packed, err := PackSgemmNTWeights(weights, n, k, k)
			if err != nil {
				b.Fatal(err)
			}
			scratch := make([]float32, k*gebpNR)
			b.Run("packed-old", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					clear(c)
					if !SgemmNTPackedTo(c, a, weights, scratch, m, n, k, 1, k, k, n) {
						b.Fatal("SgemmNTPackedTo rejected projection")
					}
				}
			})
			b.Run("prepacked", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					clear(c)
					if !SgemmNTPrepackedTo(c, a, weights, packed, m, n, k, 1, k, k, n) {
						b.Fatal("SgemmNTPrepackedTo rejected projection")
					}
				}
			})
		})
	}
}

func TestPrepackedRejectsPackedOperandOverlap(t *testing.T) {
	const m, n, k = 6, 16, 16
	raw := make([]float32, n*k)
	for _, operand := range []string{"c", "a"} {
		storage := make([]float32, 512)
		for i := range storage {
			storage[i] = float32(i + 1)
		}
		packed := storage[:256]
		a, c := make([]float32, m*k), make([]float32, m*n)
		if operand == "c" {
			c = storage[1 : 1+m*n]
		} else {
			a = storage[1 : 1+m*k]
		}
		if SgemmNTPrepackedTo(c, a, raw, packed, m, n, k, 1, k, k, n) {
			t.Fatal("accepted packed alias", operand)
		}
		for i, v := range storage {
			if v != float32(i+1) {
				t.Fatal("mutated rejected input")
			}
		}
	}
}
