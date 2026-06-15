package nvidia

import (
	"encoding/binary"
	"math"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/half"
)

func TestGateUpGELUQ4KBatchToBuffer(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	inDim, inter := 256, 2
	outDim := inter * 2
	raw := make([]byte, outDim*144)
	for r := 0; r < outDim; r++ {
		blk := raw[r*144 : (r+1)*144]
		binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(0.04+float32(r)*0.01))
		binary.LittleEndian.PutUint16(blk[2:4], half.F32ToF16(0.01))
		for i := 0; i < 12; i++ {
			blk[4+i] = byte(1 + (i+r)%10)
		}
		for i := 0; i < 128; i++ {
			blk[16+i] = byte((i*3 + r) & 0xff)
		}
	}
	x := make([]float32, inDim)
	for i := range x {
		x[i] = float32((i%13)-6) * 0.03
	}
	m, err := UploadQ4KMatrixRows(raw, inDim, outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	xb, err := Malloc(inDim)
	if err != nil {
		t.Fatal(err)
	}
	defer xb.Free()
	ob, err := Malloc(inter)
	if err != nil {
		t.Fatal(err)
	}
	defer ob.Free()
	if err := xb.Upload(x); err != nil {
		t.Fatal(err)
	}
	if err := GateUpGELUQ4KBatchToBuffer(ob, xb, 1, inter, m); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, inter)
	if err := ob.Download(got); err != nil {
		t.Fatal(err)
	}
	all := make([]float32, outDim)
	if err := GemvQ4K(all, x, m); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, inter)
	if !simd.GELUExactMulTo(want, all[:inter], all[inter:]) {
		t.Fatal("cpu gelu")
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 2e-3 {
			t.Fatalf("i=%d got=%g want=%g all=%v", i, got[i], want[i], got)
		}
	}
}
