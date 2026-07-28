package nvidia

import (
	simdfp8 "github.com/rcarmo/go-pherence/backends/simd/quant/fp8"
	"math"
	"testing"
)

func makeFP8Live(t testing.TB, outDim, inDim int) (*GPUFP8E4M3Linear, []byte, []float32) {
	w := make([]byte, outDim*inDim)
	for i := range w {
		w[i] = byte(i % 113)
	}
	s := make([]float32, outDim)
	for i := range s {
		s[i] = .01 + float32(i%7)*.001
	}
	g, e := UploadFP8E4M3Linear(w, s, nil, outDim, inDim)
	if e != nil {
		t.Fatal(e)
	}
	return g, w, s
}
func TestFP8TiledLiveParity(t *testing.T) {
	if !SgemmReady() {
		t.Skip()
	}
	const inD, outD = 1536, 2048
	g, w, s := makeFP8Live(t, outD, inD)
	defer g.Free()
	for _, batch := range []int{4, 8} {
		x := make([]float32, batch*inD)
		for i := range x {
			x[i] = float32(i%17-8) / 17
		}
		got := make([]float32, batch*outD)
		if e := GemmFP8E4M3(got, x, batch, g); e != nil {
			t.Fatal(e)
		}
		lin := simdfp8.Linear{Weight: w, Scale: s, InDim: inD, OutDim: outD}
		want := make([]float32, len(got))
		if e := lin.BatchGemvTo(x, want, batch); e != nil {
			t.Fatal(e)
		}
		mx := 0.0
		for i := range got {
			d := math.Abs(float64(got[i] - want[i]))
			if d > mx {
				mx = d
			}
		}
		if mx > .02 {
			t.Fatalf("batch=%d max=%g", batch, mx)
		}
		t.Logf("batch=%d max=%g", batch, mx)
	}
}
