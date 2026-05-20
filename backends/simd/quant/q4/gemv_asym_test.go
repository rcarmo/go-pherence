package q4

import "testing"

func packQ4(vals ...int32) int32 {
	var p int32
	for i, v := range vals {
		p |= (v & 0xF) << (uint(i) * 4)
	}
	return p
}

func TestDequantAsymParallelMatchesSerialReference(t *testing.T) {
	inDim, outDim := 8, 1024
	qweight := make([]int32, outDim)
	for j := 0; j < outDim; j++ {
		qweight[j] = packQ4(int32(j%16), int32((j+1)%16), int32((j+2)%16), int32((j+3)%16), int32((j+4)%16), int32((j+5)%16), int32((j+6)%16), int32((j+7)%16))
	}
	qzeros := make([]int32, outDim/8)
	for i := range qzeros {
		qzeros[i] = packQ4(1, 2, 3, 4, 5, 6, 7, 8)
	}
	gIdx := make([]int32, inDim)
	scales := make([]float32, outDim)
	for i := range scales {
		scales[i] = float32((i%7)+1) * 0.125
	}
	got := make([]float32, inDim*outDim+1)
	got[len(got)-1] = 123
	if !DequantTo(got, qweight, qzeros, gIdx, scales, inDim, outDim, false) {
		t.Fatal("DequantTo returned false for valid asymmetric input")
	}
	for j := 0; j < outDim; j++ {
		for i := 0; i < inDim; i++ {
			packIdx := i / 8
			qw := (qweight[packIdx*outDim+j] >> (uint(i%8) * 4)) & 0xF
			qz := (qzeros[j/8] >> (uint(j%8) * 4)) & 0xF
			want := scales[j] * float32(qw-qz)
			if got[j*inDim+i] != want {
				t.Fatalf("got[%d,%d]=%g want %g", j, i, got[j*inDim+i], want)
			}
		}
	}
	if got[len(got)-1] != 123 {
		t.Fatal("DequantTo mutated tail")
	}
}

func TestGemvAsymMatchesDequantReference(t *testing.T) {
	inDim, outDim := 8, 8
	x := []float32{1, -2, 3, -4, 0.5, -0.25, 2, -1}
	qweight := make([]int32, outDim)
	for j := 0; j < outDim; j++ {
		qweight[j] = packQ4(int32(j%16), int32((j+1)%16), int32((j+2)%16), int32((j+3)%16), int32((j+4)%16), int32((j+5)%16), int32((j+6)%16), int32((j+7)%16))
	}
	qzeros := []int32{packQ4(1, 2, 3, 4, 5, 6, 7, 8)}
	gIdx := make([]int32, inDim)
	scales := []float32{0.5, -0.25, 1, 1.5, -1, 0.75, 2, -2}
	wantW := Dequant(qweight, qzeros, gIdx, scales, inDim, outDim, false)
	if wantW == nil {
		t.Fatal("Dequant returned nil")
	}
	want := make([]float32, outDim)
	for j := 0; j < outDim; j++ {
		for i := 0; i < inDim; i++ {
			want[j] += x[i] * wantW[j*inDim+i]
		}
	}
	got := make([]float32, outDim)
	if !GemvTo(got, x, qweight, qzeros, gIdx, scales, inDim, outDim, false) {
		t.Fatal("GemvTo returned false for valid asymmetric input")
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("out[%d]=%g want %g", i, got[i], want[i])
		}
	}
}

func TestGemvToRejectsMalformedInputs(t *testing.T) {
	out, x, qw, g, s, inDim, outDim := validGemvQ4Inputs()
	if !GemvSymTo(out, x, qw, g, s, inDim, outDim) {
		t.Fatal("GemvSymTo returned false for valid symmetric input")
	}
	if GemvSymTo(out[:1], x, qw, g, s, inDim, outDim) {
		t.Fatal("GemvSymTo accepted short output")
	}
	if GemvTo(out[:1], x, qw, nil, g, s, inDim, outDim, true) {
		t.Fatal("GemvTo accepted short output")
	}
}

func TestValidateGemvAsymRejectsMissingQZeros(t *testing.T) {
	out, x, qw, g, s, inDim, _ := validGemvQ4Inputs()
	outDim := 8
	qw = make([]int32, (inDim/8)*outDim)
	s = make([]float32, outDim)
	if err := ValidateGemv(out, x, qw, nil, g, s, inDim, outDim, false); err == nil {
		t.Fatal("expected missing qzeros error")
	}
}
