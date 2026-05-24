package fft

import (
	"math"
	"testing"
)

func TestSinusoidalPosEmbed(t *testing.T) {
	pe := SinusoidalPosEmbed(100, 64)
	if len(pe) != 100*64 {
		t.Fatalf("length=%d want %d", len(pe), 100*64)
	}
	// pos=0, dim=0: sin(0)=0
	if pe[0] != 0 {
		t.Fatalf("PE[0,0]=%f want 0", pe[0])
	}
	// pos=0, dim=1: cos(0)=1
	if pe[1] != 1 {
		t.Fatalf("PE[0,1]=%f want 1", pe[1])
	}
	// pos=1, dim=0: sin(1*invFreq[0]) where invFreq[0]=1
	expected := float32(math.Sin(1.0))
	if math.Abs(float64(pe[64]-expected)) > 0.001 {
		t.Fatalf("PE[1,0]=%f want %f", pe[64], expected)
	}
}

func TestAddPosEmbed(t *testing.T) {
	dModel := 4
	seqLen := 2
	h := make([]float32, seqLen*dModel) // all zeros
	pe := SinusoidalPosEmbed(10, dModel)

	AddPosEmbed(h, pe, seqLen, dModel, 0)

	// h should now equal pe[0:seqLen*dModel]
	for i := 0; i < seqLen*dModel; i++ {
		if math.Abs(float64(h[i]-pe[i])) > 0.001 {
			t.Fatalf("h[%d]=%f want %f", i, h[i], pe[i])
		}
	}
}

func BenchmarkSinusoidalPosEmbed3000x1280(b *testing.B) {
	for i := 0; i < b.N; i++ {
		SinusoidalPosEmbed(3000, 1280)
	}
}
