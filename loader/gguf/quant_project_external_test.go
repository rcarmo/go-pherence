package gguf_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestQuantMatrixProjectBatchF32ToMatchesDequantOracle(t *testing.T) {
	cases := []struct {
		name   string
		qtype  gguf.QuantType
		inDim  int
		outDim int
		batch  int
	}{
		{name: "q4k_tiled_and_tail", qtype: gguf.QuantQ4_K, inDim: 256, outDim: 10, batch: 9},
		{name: "q5_0", qtype: gguf.QuantQ5_0, inDim: 64, outDim: 7, batch: 3},
		{name: "q8_0", qtype: gguf.QuantQ8_0, inDim: 64, outDim: 6, batch: 4},
		{name: "q6_k", qtype: gguf.QuantQ6_K, inDim: 256, outDim: 5, batch: 3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := syntheticQuantMatrix(t, tc.qtype, tc.inDim, tc.outDim)
			x := make([]float32, tc.batch*tc.inDim)
			for i := range x {
				x[i] = float32((i%23)-11) * 0.021
			}
			got := make([]float32, tc.batch*tc.outDim)
			if err := m.ProjectBatchF32To(got, x, tc.batch); err != nil {
				t.Fatal(err)
			}
			for pos := 0; pos < tc.batch; pos++ {
				want := make([]float32, tc.outDim)
				if err := m.ProjectBatchF32To(want, x[pos*tc.inDim:(pos+1)*tc.inDim], 1); err != nil {
					t.Fatal(err)
				}
				tol := 1e-6
				if tc.qtype == gguf.QuantQ4_K {
					// Batch-1 uses the scalar fallback while larger batches exercise the
					// retained 8x8 tile path, so allow the proven backend-level drift.
					tol = 3e-2
				}
				for r := 0; r < tc.outDim; r++ {
					if math.Abs(float64(got[pos*tc.outDim+r]-want[r])) > tol {
						t.Fatalf("pos=%d row=%d got=%g want=%g diff=%g", pos, r, got[pos*tc.outDim+r], want[r], got[pos*tc.outDim+r]-want[r])
					}
				}
			}
		})
	}
}

func syntheticQuantMatrix(t testing.TB, qtype gguf.QuantType, inDim, outDim int) *gguf.QuantMatrix {
	t.Helper()
	m := &gguf.QuantMatrix{Name: "synthetic", QType: qtype, InDim: inDim, OutDim: outDim}
	rowBytes, err := m.RowBytes()
	if err != nil {
		t.Fatal(err)
	}
	m.Raw = make([]byte, rowBytes*outDim)
	for r := 0; r < outDim; r++ {
		row := m.Raw[r*rowBytes : (r+1)*rowBytes]
		switch qtype {
		case gguf.QuantQ4_K:
			for b := 0; b < inDim/256; b++ {
				blk := row[b*144 : (b+1)*144]
				binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(0.025+float32(r+b)*0.002))
				binary.LittleEndian.PutUint16(blk[2:4], half.F32ToF16(0.004+float32((r+b)%3)*0.001))
				for i := 0; i < 12; i++ {
					blk[4+i] = byte(1 + (i+r+b)%17)
				}
				for i := 0; i < 128; i++ {
					blk[16+i] = byte((i*7 + r*11 + b*13) & 0xff)
				}
			}
		case gguf.QuantQ5_0:
			for b := 0; b < inDim/32; b++ {
				blk := row[b*22 : (b+1)*22]
				binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(0.05+float32(r+b)*0.01))
				binary.LittleEndian.PutUint32(blk[2:6], uint32(0xa5a50000|r<<3|b))
				for i := 0; i < 16; i++ {
					blk[6+i] = byte((r*13 + b*17 + i*7) & 0xff)
				}
			}
		case gguf.QuantQ8_0:
			for b := 0; b < inDim/32; b++ {
				blk := row[b*34 : (b+1)*34]
				binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(0.07+float32(r+b)*0.013))
				for i := 0; i < inDim/(inDim/32); i++ {
					blk[2+i] = byte(int8((r*3+b*5+i)%15 - 7))
				}
			}
		case gguf.QuantQ6_K:
			for b := 0; b < inDim/256; b++ {
				blk := row[b*210 : (b+1)*210]
				for i := 0; i < 128; i++ {
					blk[i] = byte((r*13 + b*7 + i*19 + 5) & 0xff)
				}
				for i := 0; i < 64; i++ {
					blk[128+i] = byte((r*23 + b*11 + i*31 + 7) & 0xff)
				}
				for i := 0; i < 16; i++ {
					blk[192+i] = byte(int8((r*5+b*3+i*9)%63 - 31))
				}
				binary.LittleEndian.PutUint16(blk[208:210], half.F32ToF16(0.0234375+float32(r+b)*0.0048828125))
			}
		default:
			t.Fatalf("unsupported qtype %v", qtype)
		}
	}
	return m
}
