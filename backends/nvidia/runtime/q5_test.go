package nvidia

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestGemvQ5_0BatchMatchesCPU(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	inDim, outDim, batch := 64, 5, 3
	rowBytes := (inDim / 32) * 22
	raw := make([]byte, outDim*rowBytes)
	for r := 0; r < outDim; r++ {
		for b := 0; b < inDim/32; b++ {
			blk := raw[r*rowBytes+b*22:]
			binary.LittleEndian.PutUint16(blk[:2], half.F32ToF16(0.025+float32(r+b)*0.003))
			var high uint32
			for i := 0; i < 32; i++ {
				q := (i*3 + r*5 + b*7) % 32
				if q >= 16 {
					high |= 1 << uint(i)
				}
				if i < 16 {
					blk[6+i] = (blk[6+i] & 0xF0) | byte(q&0x0F)
				} else {
					blk[6+i-16] = (blk[6+i-16] & 0x0F) | byte((q&0x0F)<<4)
				}
			}
			binary.LittleEndian.PutUint32(blk[2:6], high)
		}
	}
	m, err := UploadQ5_0MatrixRows(raw, inDim, outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	x := make([]float32, batch*inDim)
	for i := range x {
		x[i] = float32((i%17)-8) * 0.021
	}
	got := make([]float32, batch*outDim)
	if err := GemvQ5_0Batch(got, x, batch, m); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < outDim; r++ {
		deq := dequantQ5_0Test(raw[r*rowBytes:(r+1)*rowBytes], inDim)
		for b := 0; b < batch; b++ {
			var want float32
			xrow := x[b*inDim : (b+1)*inDim]
			for i := range deq {
				want += deq[i] * xrow[i]
			}
			if math.Abs(float64(got[b*outDim+r]-want)) > 1e-4 {
				t.Fatalf("batch=%d row=%d got=%g want=%g all=%v", b, r, got[b*outDim+r], want, got)
			}
		}
	}
}

func dequantQ5_0Test(raw []byte, n int) []float32 {
	out := make([]float32, n)
	for b := 0; b < n/32; b++ {
		blk := raw[b*22:]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[:2]))
		high := binary.LittleEndian.Uint32(blk[2:6])
		qs := blk[6:22]
		for i := 0; i < 16; i++ {
			q0 := int(qs[i] & 0x0F)
			q1 := int(qs[i] >> 4)
			if high&(1<<uint(i)) != 0 {
				q0 |= 16
			}
			if high&(1<<uint(i+16)) != 0 {
				q1 |= 16
			}
			out[b*32+i] = d * float32(q0-16)
			out[b*32+i+16] = d * float32(q1-16)
		}
	}
	return out
}
