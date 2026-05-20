package format

import "testing"

func packNibbles(vals ...int32) int32 {
	var out uint32
	for i, v := range vals {
		out |= uint32(v&0xf) << (uint(i) * 4)
	}
	return int32(out)
}

func TestDequantAsymmetricKnownValues(t *testing.T) {
	const inFeatures, outFeatures = 8, 8
	qweight := make([]int32, outFeatures)
	for j := 0; j < outFeatures; j++ {
		qweight[j] = packNibbles(int32(j), int32(j+1), int32(j+2), int32(j+3), int32(j+4), int32(j+5), int32(j+6), int32(j+7))
	}
	gIdx := []int32{0, 0, 0, 0, 1, 1, 1, 1}
	scales := []float32{
		1, 2, 3, 4, 5, 6, 7, 8,
		0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
	}
	qzeros := []int32{
		packNibbles(0, 1, 2, 3, 4, 5, 6, 7),
		packNibbles(8, 7, 6, 5, 4, 3, 2, 1),
	}

	got := Dequant(qweight, qzeros, gIdx, scales, inFeatures, outFeatures, false)
	if len(got) != inFeatures*outFeatures {
		t.Fatalf("len=%d", len(got))
	}
	gotTo := make([]float32, len(got)+1)
	gotTo[len(got)] = 123
	if !DequantTo(gotTo, qweight, qzeros, gIdx, scales, inFeatures, outFeatures, false) {
		t.Fatal("DequantTo returned false for valid inputs")
	}
	if gotTo[len(got)] != 123 {
		t.Fatalf("DequantTo mutated tail: %v", gotTo)
	}
	for j := 0; j < outFeatures; j++ {
		for i := 0; i < inFeatures; i++ {
			g := int(gIdx[i])
			qw := int32(j + i)
			var qz int32
			if g == 0 {
				qz = int32(j)
			} else {
				qz = int32(8 - j)
			}
			want := scales[g*outFeatures+j] * float32(qw-qz)
			if got[j*inFeatures+i] != want {
				t.Fatalf("out[%d,%d]=%g want %g", j, i, got[j*inFeatures+i], want)
			}
			if gotTo[j*inFeatures+i] != want {
				t.Fatalf("DequantTo out[%d,%d]=%g want %g", j, i, gotTo[j*inFeatures+i], want)
			}
		}
	}
}

func TestGemvSymMatchesDequantSymReference(t *testing.T) {
	const inDim, outDim = 16, 5
	x := []float32{1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16}
	qweight := make([]int32, (inDim/8)*outDim)
	for pack := 0; pack < inDim/8; pack++ {
		for j := 0; j < outDim; j++ {
			vals := make([]int32, 8)
			for bit := 0; bit < 8; bit++ {
				vals[bit] = int32((pack*3 + j*5 + bit) & 0xf)
			}
			qweight[pack*outDim+j] = packNibbles(vals...)
		}
	}
	gIdx := []int32{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 0, 0}
	scales := []float32{
		0.25, 0.5, 0.75, 1, 1.25,
		1.5, 1.75, 2, 2.25, 2.5,
		-0.5, -0.25, 0.125, 0.375, 0.625,
	}
	out := []float32{99, 99, 99, 99, 99, 123}
	GemvSym(out, x, qweight, gIdx, scales, inDim, outDim)
	deq := DequantSym(qweight, gIdx, scales, inDim, outDim)
	deqTo := make([]float32, len(deq)+1)
	deqTo[len(deq)] = 456
	if !DequantSymTo(deqTo, qweight, gIdx, scales, inDim, outDim) {
		t.Fatal("DequantSymTo returned false for valid inputs")
	}
	if deqTo[len(deq)] != 456 {
		t.Fatalf("DequantSymTo mutated tail: %v", deqTo)
	}
	want := make([]float32, outDim)
	for j := 0; j < outDim; j++ {
		for i := 0; i < inDim; i++ {
			if deqTo[j*inDim+i] != deq[j*inDim+i] {
				t.Fatalf("DequantSymTo[%d,%d]=%g want %g", j, i, deqTo[j*inDim+i], deq[j*inDim+i])
			}
			want[j] += deq[j*inDim+i] * x[i]
		}
	}
	for j := 0; j < outDim; j++ {
		if out[j] != want[j] {
			t.Fatalf("out[%d]=%g want %g (all=%v)", j, out[j], want[j], out)
		}
	}
	if out[outDim] != 123 {
		t.Fatalf("GemvSym mutated tail value: %v", out)
	}
}

func TestGemvSymKnownValues(t *testing.T) {
	const inDim, outDim = 8, 2
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	qweight := []int32{
		packNibbles(8, 9, 10, 11, 12, 13, 14, 15),
		packNibbles(7, 6, 5, 4, 3, 2, 1, 0),
	}
	gIdx := []int32{0, 0, 0, 0, 1, 1, 1, 1}
	scales := []float32{1, 2, 10, 20}
	out := []float32{99, 99}
	GemvSym(out, x, qweight, gIdx, scales, inDim, outDim)

	want0 := float32(0)
	want1 := float32(0)
	for i := 0; i < inDim; i++ {
		g := int(gIdx[i])
		want0 += x[i] * scales[g*outDim+0] * float32(i)
		want1 += x[i] * scales[g*outDim+1] * float32(-1-i)
	}
	if out[0] != want0 || out[1] != want1 {
		t.Fatalf("GemvSym=%v want [%g %g]", out, want0, want1)
	}
}
