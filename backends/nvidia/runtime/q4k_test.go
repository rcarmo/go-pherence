package nvidia

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestGemvQ4KMatchesCPU(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	inDim, outDim := 256, 2
	raw := make([]byte, outDim*144)
	for r := 0; r < outDim; r++ {
		blk := raw[r*144 : (r+1)*144]
		binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(0.1+float32(r)*0.03))
		binary.LittleEndian.PutUint16(blk[2:4], half.F32ToF16(0.02))
		for i := 0; i < 12; i++ {
			blk[4+i] = byte(1 + (i+r)%12)
		}
		for i := 0; i < 128; i++ {
			blk[16+i] = byte((i + r*3) & 0xff)
		}
	}
	x := make([]float32, inDim)
	for i := range x {
		x[i] = float32((i%11)-5) * 0.05
	}
	m, err := UploadQ4KMatrixRows(raw, inDim, outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	got := make([]float32, outDim)
	if err := GemvQ4K(got, x, m); err != nil {
		t.Fatal(err)
	}
	gotBatch := make([]float32, outDim*2)
	if err := GemvQ4KBatch(gotBatch, append(x, x...), 2, m); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < outDim; r++ {
		row := raw[r*144 : (r+1)*144]
		deq := dequantQ4KTest(row, inDim)
		var want float32
		for i := range deq {
			want += deq[i] * x[i]
		}
		if math.Abs(float64(got[r]-want)) > 1e-3 {
			t.Fatalf("row=%d got=%g want=%g all=%v", r, got[r], want, got)
		}
		if math.Abs(float64(gotBatch[r]-want)) > 1e-3 || math.Abs(float64(gotBatch[outDim+r]-want)) > 1e-3 {
			t.Fatalf("batch row=%d got=%g/%g want=%g all=%v", r, gotBatch[r], gotBatch[outDim+r], want, gotBatch)
		}
	}
}

func dequantQ4KTest(raw []byte, n int) []float32 {
	out := make([]float32, n)
	for b := 0; b < n/256; b++ {
		blk := raw[b*144:]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[0:2]))
		dmin := half.F16ToF32(binary.LittleEndian.Uint16(blk[2:4]))
		sc := blk[4:16]
		qs := blk[16:144]
		var scales [8]float32
		var mins [8]float32
		for j := 0; j < 4; j++ {
			scales[j] = float32(sc[j]&63) * d
			mins[j] = float32(sc[j+4]&63) * dmin
		}
		for j := 4; j < 8; j++ {
			k := j - 4
			scales[j] = float32((sc[j+4]&0xF)|((sc[k]>>6)<<4)) * d
			mins[j] = float32((sc[j+4]>>4)|((sc[k+4]>>6)<<4)) * dmin
		}
		base := b * 256
		for group := 0; group < 8; group++ {
			q := qs[(group/2)*32:]
			for i := 0; i < 16; i++ {
				var q0, q1 int
				if group%2 == 0 {
					q0 = int(q[i] & 0x0f)
					q1 = int(q[i+16] & 0x0f)
				} else {
					q0 = int(q[i] >> 4)
					q1 = int(q[i+16] >> 4)
				}
				out[base+group*32+i] = scales[group]*float32(q0) - mins[group]
				out[base+group*32+16+i] = scales[group]*float32(q1) - mins[group]
			}
		}
	}
	return out
}
