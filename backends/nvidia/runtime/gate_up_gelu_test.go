package nvidia

import (
	"math"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func TestGateUpGELUBuffer(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	batch, inter := 2, 4
	src := []float32{0.1, -0.2, 0.3, -0.4, 1, 2, 3, 4, 0.5, -0.6, 0.7, -0.8, -1, 1.5, -2, 2.5}
	srcBuf, err := Malloc(len(src))
	if err != nil {
		t.Fatal(err)
	}
	defer srcBuf.Free()
	outBuf, err := Malloc(batch * inter)
	if err != nil {
		t.Fatal(err)
	}
	defer outBuf.Free()
	if err := srcBuf.Upload(src); err != nil {
		t.Fatal(err)
	}
	if err := GateUpGELUBuffer(srcBuf, outBuf, batch, inter); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, batch*inter)
	if err := outBuf.Download(got); err != nil {
		t.Fatal(err)
	}
	for b := 0; b < batch; b++ {
		want := make([]float32, inter)
		gate := src[b*2*inter : b*2*inter+inter]
		up := src[b*2*inter+inter : (b+1)*2*inter]
		if !simd.GELUExactMulTo(want, gate, up) {
			t.Fatal("cpu gelu failed")
		}
		for i := range want {
			if math.Abs(float64(got[b*inter+i]-want[i])) > 2e-3 {
				t.Fatalf("b=%d i=%d got=%g want=%g all=%v", b, i, got[b*inter+i], want[i], got)
			}
		}
	}
}
