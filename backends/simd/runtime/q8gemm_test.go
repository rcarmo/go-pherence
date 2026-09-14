package simd

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
	"unsafe"

	"github.com/rcarmo/go-pherence/half"
)

func TestSgemmNTQ8_0ToMatchesReference(t *testing.T) {
	for _, tc := range []struct {
		name    string
		m, n, k int
	}{
		{name: "single", m: 1, n: 1, k: 32},
		{name: "row-tail-3", m: 3, n: 5, k: 64},
		{name: "tile-4", m: 4, n: 7, k: 96},
		{name: "row-tail-1", m: 5, n: 3, k: 64},
		{name: "row-tail-3-large", m: 7, n: 9, k: 128},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := makeTestQ8GemmA(tc.m, tc.k)
			w := makeTestQ8_0Raw(tc.n, tc.k)
			want := refSgemmNTQ8_0(a, w, tc.m, tc.n, tc.k)
			got := make([]float32, tc.m*tc.n)
			for i := range got {
				got[i] = -1234
			}
			if !SgemmNTQ8_0To(got, a, w, tc.m, tc.n, tc.k) {
				t.Fatal("SgemmNTQ8_0To rejected valid inputs")
			}
			for i := range want {
				if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-4 {
					t.Fatalf("idx=%d got=%g want=%g diff=%g", i, got[i], want[i], diff)
				}
			}
		})
	}
}

func TestSgemmNTQ8_0ToOverwriteSemantics(t *testing.T) {
	const (
		m = 5
		n = 4
		k = 64
	)
	a := makeTestQ8GemmA(m, k)
	w := makeTestQ8_0Raw(n, k)
	want := refSgemmNTQ8_0(a, w, m, n, k)
	got := make([]float32, m*n)
	for i := range got {
		got[i] = float32(i+1) * 17
	}
	if !SgemmNTQ8_0To(got, a, w, m, n, k) {
		t.Fatal("SgemmNTQ8_0To rejected valid inputs")
	}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-4 {
			t.Fatalf("idx=%d got=%g want=%g", i, got[i], want[i])
		}
	}
}

func TestSgemmNTQ8_0ToNoAllocs(t *testing.T) {
	const (
		m = 7
		n = 17
		k = 128
	)
	a := makeTestQ8GemmA(m, k)
	w := makeTestQ8_0Raw(n, k)
	c := make([]float32, m*n)
	if allocs := testing.AllocsPerRun(20, func() {
		if !SgemmNTQ8_0To(c, a, w, m, n, k) {
			panic("SgemmNTQ8_0To returned false")
		}
	}); allocs != 0 {
		t.Fatalf("allocs %g", allocs)
	}
}

func TestSgemmNTQ8_0ToRejectsInvalidAndDoesNotMutate(t *testing.T) {
	const (
		m = 2
		n = 3
		k = 32
	)
	validA := makeTestQ8GemmA(m, k)
	validW := makeTestQ8_0Raw(n, k)
	validC := make([]float32, m*n)
	for i := range validC {
		validC[i] = float32(i+1) * 0.5
	}

	cases := []struct {
		name string
		mk   func() ([]float32, []float32, []byte, int, int, int)
	}{
		{
			name: "zero-m",
			mk: func() ([]float32, []float32, []byte, int, int, int) {
				return append([]float32(nil), validC...), append([]float32(nil), validA...), append([]byte(nil), validW...), 0, n, k
			},
		},
		{
			name: "k-not-multiple-32",
			mk: func() ([]float32, []float32, []byte, int, int, int) {
				return append([]float32(nil), validC...), append([]float32(nil), validA...), append([]byte(nil), validW...), m, n, k - 1
			},
		},
		{
			name: "short-a",
			mk: func() ([]float32, []float32, []byte, int, int, int) {
				return append([]float32(nil), validC...), append([]float32(nil), validA[:len(validA)-1]...), append([]byte(nil), validW...), m, n, k
			},
		},
		{
			name: "short-c",
			mk: func() ([]float32, []float32, []byte, int, int, int) {
				return append([]float32(nil), validC[:len(validC)-1]...), append([]float32(nil), validA...), append([]byte(nil), validW...), m, n, k
			},
		},
		{
			name: "short-w",
			mk: func() ([]float32, []float32, []byte, int, int, int) {
				return append([]float32(nil), validC...), append([]float32(nil), validA...), append([]byte(nil), validW[:len(validW)-1]...), m, n, k
			},
		},
		{
			name: "non-finite-scale",
			mk: func() ([]float32, []float32, []byte, int, int, int) {
				w := append([]byte(nil), validW...)
				binary.LittleEndian.PutUint16(w[:2], 0x7e00)
				return append([]float32(nil), validC...), append([]float32(nil), validA...), w, m, n, k
			},
		},
		{
			name: "c-overlaps-a",
			mk: func() ([]float32, []float32, []byte, int, int, int) {
				backing := make([]float32, m*k+m*n)
				for i := range backing {
					backing[i] = float32(i + 1)
				}
				a := backing[:m*k]
				c := backing[k : k+m*n]
				return c, a, append([]byte(nil), validW...), m, n, k
			},
		},
		{
			name: "c-overlaps-w-bytes",
			mk: func() ([]float32, []float32, []byte, int, int, int) {
				storage := make([]float32, 32)
				for i := range storage {
					storage[i] = float32(i + 1)
				}
				c := storage[:m*n]
				wb := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(storage))), len(storage)*4)
				w := wb[4 : 4+len(validW)]
				return c, append([]float32(nil), validA...), w, m, n, k
			},
		},
		{
			name: "overflow",
			mk: func() ([]float32, []float32, []byte, int, int, int) {
				maxInt := int(^uint(0) >> 1)
				return append([]float32(nil), validC...), append([]float32(nil), validA...), append([]byte(nil), validW...), maxInt, 2, 32
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, a, w, m0, n0, k0 := tc.mk()
			beforeC := append([]float32(nil), c...)
			beforeA := append([]float32(nil), a...)
			beforeW := append([]byte(nil), w...)
			if SgemmNTQ8_0To(c, a, w, m0, n0, k0) {
				t.Fatal("accepted malformed inputs")
			}
			if !equalFloat32Bits(c, beforeC) {
				t.Fatal("mutated c on rejected input")
			}
			if !equalFloat32Bits(a, beforeA) {
				t.Fatal("mutated a on rejected input")
			}
			if !bytes.Equal(w, beforeW) {
				t.Fatal("mutated w on rejected input")
			}
		})
	}
}

func makeTestQ8GemmA(m, k int) []float32 {
	a := make([]float32, m*k)
	for i := 0; i < m; i++ {
		for p := 0; p < k; p++ {
			a[i*k+p] = float32(((i+3)*(p%11) - 17 + p%5)) * 0.03125
		}
	}
	return a
}

func makeTestQ8_0Raw(n, k int) []byte {
	blocks := k / q8_0BlockElems
	rowBytes := blocks * q8_0BlockBytes
	w := make([]byte, n*rowBytes)
	for j := 0; j < n; j++ {
		row := w[j*rowBytes : (j+1)*rowBytes]
		for b := 0; b < blocks; b++ {
			blk := row[b*q8_0BlockBytes : (b+1)*q8_0BlockBytes]
			scale := float32(0.015625 * float32(1+(j+b)%9))
			binary.LittleEndian.PutUint16(blk[:2], half.F32ToF16(scale))
			for i := 0; i < q8_0BlockElems; i++ {
				blk[2+i] = byte(int8(((j+1)*(i%13)+b*3)%23 - 11))
			}
		}
	}
	return w
}

func refSgemmNTQ8_0(a []float32, w []byte, m, n, k int) []float32 {
	blocks := k / q8_0BlockElems
	rowBytes := blocks * q8_0BlockBytes
	out := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			row := w[j*rowBytes : (j+1)*rowBytes]
			var sum float32
			for b := 0; b < blocks; b++ {
				blk := row[b*q8_0BlockBytes : (b+1)*q8_0BlockBytes]
				d := half.F16ToF32(binary.LittleEndian.Uint16(blk[:2]))
				x := a[i*k+b*q8_0BlockElems:]
				for p := 0; p < q8_0BlockElems; p++ {
					sum += d * float32(int8(blk[2+p])) * x[p]
				}
			}
			out[i*n+j] = sum
		}
	}
	return out
}

func equalFloat32Bits(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}
