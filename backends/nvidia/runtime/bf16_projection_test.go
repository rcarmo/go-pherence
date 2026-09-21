package nvidia

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestBF16ProjectionRejectsInvalid(t *testing.T) {
	for _, shape := range [][2]int{{0, 2}, {2, -1}, {int(^uint(0) >> 1), 2}, {4, 4}} {
		if err := WidenBF16Transpose(nil, nil, shape[0], shape[1]); err == nil {
			t.Fatal("accepted invalid inputs", shape)
		}
	}
}

func TestBF16ProjectionGPU(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_BF16_PROJECTION") != "1" || os.Getenv("GO_PHERENCE_DISABLE_NVIDIA") != "" {
		t.Skip("opt-in GPU projection fixture")
	}
	if !Available() {
		t.Fatal("GPU requested but unavailable")
	}
	for _, shape := range [][2]int{{1, 1}, {3, 5}, {64, 65}, {257, 64}} {
		r, c := shape[0], shape[1]
		n := r * c
		raw := make([]byte, n*2)
		want := make([]float32, n)
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				v := float32(math.Sin(float64(i*c+j) * 0.17))
				bits := uint16(math.Float32bits(v) >> 16)
				binary.LittleEndian.PutUint16(raw[(i*c+j)*2:], bits)
				want[j*r+i] = half.BF16ToF32(bits)
			}
		}
		src, err := Malloc((len(raw) + 3) / 4)
		if err != nil {
			t.Fatal(err)
		}
		dst, err := Malloc(n)
		if err != nil {
			src.Free()
			t.Fatal(err)
		}
		if err = src.UploadBytes(raw); err != nil {
			t.Fatal(err)
		}
		if err = WidenBF16Transpose(dst, src, r, c); err != nil {
			t.Fatal(err)
		}
		if err = SyncErr(); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, n)
		if err = dst.Download(got); err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%v element %d got=%g want=%g", shape, i, got[i], want[i])
			}
		}
		src.Free()
		dst.Free()
	}
}

func TestCompensatedProjectionGPU(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_BF16_PROJECTION") != "1" || os.Getenv("GO_PHERENCE_DISABLE_NVIDIA") != "" {
		t.Skip("opt-in GPU test")
	}
	if !Available() {
		t.Fatal("GPU unavailable")
	}
	for _, shape := range [][3]int{{1, 3, 65}, {3, 17, 128}, {17, 19, 1024}} {
		m, n, k := shape[0], shape[1], shape[2]
		a, b := make([]float32, m*k), make([]float32, k*n)
		for i := range a {
			a[i] = float32(math.Sin(float64(i) * 0.13))
		}
		for i := range b {
			b[i] = float32(math.Cos(float64(i) * 0.29))
		}
		ab, err := Malloc(len(a))
		if err != nil {
			t.Fatal(err)
		}
		bb, err := Malloc(len(b))
		if err != nil {
			t.Fatal(err)
		}
		cb, err := Malloc(m * n)
		if err != nil {
			t.Fatal(err)
		}
		if err = ab.Upload(a); err != nil {
			t.Fatal(err)
		}
		if err = bb.Upload(b); err != nil {
			t.Fatal(err)
		}
		if err = SgemmCompensated(m, n, k, 1, ab, bb, cb); err != nil {
			t.Fatal(err)
		}
		if err = SyncErr(); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, m*n)
		if err = cb.Download(got); err != nil {
			t.Fatal(err)
		}
		for r := 0; r < m; r++ {
			for c := 0; c < n; c++ {
				var want float64
				for j := 0; j < k; j++ {
					want += float64(a[r*k+j]) * float64(b[j*n+c])
				}
				if math.Abs(float64(got[r*n+c])-want) > 1e-5*(1+math.Abs(want)) {
					t.Fatalf("shape=%v [%d,%d] got=%g want=%g", shape, r, c, got[r*n+c], want)
				}
			}
		}
		ab.Free()
		bb.Free()
		cb.Free()
	}
}
