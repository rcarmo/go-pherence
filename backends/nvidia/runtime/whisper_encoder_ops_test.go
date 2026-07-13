package nvidia

import (
	"math"
	"testing"
)

func TestWhisperEncoderResidentOps(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	rows, cols := 3, 5
	x := []float32{-2, -1, 0, 1, 2, 0.5, -0.5, 3, -3, 0.25, 1.5, 2.5, -1.5, -2.5, 4}
	weight := []float32{1, 2, 3, 4, 5}
	bias := []float32{0.1, 0.2, 0.3, 0.4, 0.5}
	xb, wb, bb := NewDevBufFrom(x), NewDevBufFrom(weight), NewDevBufFrom(bias)
	for _, b := range []*DevBuf{xb, wb, bb} {
		if err := b.ToGPU(); err != nil {
			t.Fatal(err)
		}
		defer b.Free()
	}
	out, err := NewDevBufGPU(len(x))
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()
	if err := WhisperRowAffineBuffer(out.GPUBuffer(), xb.GPUBuffer(), wb.GPUBuffer(), bb.GPUBuffer(), rows, cols); err != nil {
		t.Fatal(err)
	}
	got := out.Data()
	for i, value := range got {
		want := x[i]*weight[i%cols] + bias[i%cols]
		if math.Abs(float64(value-want)) > 1e-6 {
			t.Fatalf("affine[%d]=%g want %g", i, value, want)
		}
	}

	transpose, err := NewDevBufGPU(len(x))
	if err != nil {
		t.Fatal(err)
	}
	defer transpose.Free()
	if err := out.ToGPU(); err != nil {
		t.Fatal(err)
	}
	if err := WhisperTransposeBuffer(transpose.GPUBuffer(), out.GPUBuffer(), rows, cols); err != nil {
		t.Fatal(err)
	}
	gotT := transpose.Data()
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			if gotT[c*rows+r] != got[r*cols+c] {
				t.Fatalf("transpose[%d,%d]=%g want %g", c, r, gotT[c*rows+r], got[r*cols+c])
			}
		}
	}

	gelu := NewDevBufFrom(append([]float32(nil), x...))
	if err := gelu.ToGPU(); err != nil {
		t.Fatal(err)
	}
	defer gelu.Free()
	if err := WhisperGELUTanhBuffer(gelu.GPUBuffer(), len(x)); err != nil {
		t.Fatal(err)
	}
	for i, value := range gelu.Data() {
		v := float64(x[i])
		want := 0.5 * v * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(v+0.044715*v*v*v)))
		if math.Abs(float64(value)-want) > 2e-4 {
			t.Fatalf("gelu[%d]=%g want %g", i, value, want)
		}
	}
}

func TestWhisperEncoderResidentOpsRejectMalformed(t *testing.T) {
	if err := WhisperRowAffineBuffer(nil, nil, nil, nil, 1, 1); err == nil {
		t.Fatal("row affine accepted nil buffers")
	}
	if err := WhisperTransposeBuffer(nil, nil, 1, 1); err == nil {
		t.Fatal("transpose accepted nil buffers")
	}
	if err := WhisperGELUTanhBuffer(nil, 1); err == nil {
		t.Fatal("GELU accepted nil buffer")
	}
}
