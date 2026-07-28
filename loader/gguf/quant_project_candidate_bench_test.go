package gguf

import (
	"encoding/binary"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

// These benchmarks keep the existing Q5_0/Q8_0/Q6_K specialized paths honest
// against the shared dequant+Sdotx4 candidate. Do not switch the production
// dispatch unless this candidate is measurably faster.
func BenchmarkProjectBatchDequantSdotx4Candidates(b *testing.B) {
	cases := []struct {
		name  string
		qtype QuantType
		inDim int
		alt   func(m *QuantMatrix, dst, x []float32, batch int) error
	}{
		{
			name:  "q5_0",
			qtype: QuantQ5_0,
			inDim: 2816,
			alt: func(m *QuantMatrix, dst, x []float32, batch int) error {
				return m.projectBatchQ5_0DequantSdotx4To(dst, x, batch)
			},
		},
		{
			name:  "q8_0",
			qtype: QuantQ8_0,
			inDim: 2816,
			alt: func(m *QuantMatrix, dst, x []float32, batch int) error {
				return m.projectBatchQ8_0DequantSdotx4To(dst, x, batch)
			},
		},
		{
			name:  "q6_k",
			qtype: QuantQ6_K,
			inDim: 2816,
			alt: func(m *QuantMatrix, dst, x []float32, batch int) error {
				return m.projectBatchQ6KDequantSdotx4To(dst, x, batch)
			},
		},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			const (
				outDim = 1024
				batch  = 8
			)
			m := syntheticProjectBenchMatrix(b, tc.qtype, tc.inDim, outDim)
			x := make([]float32, batch*tc.inDim)
			for i := range x {
				x[i] = float32((i%31)-15) * 0.01
			}
			dst := make([]float32, batch*outDim)
			b.Run("current_special_path", func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					if err := m.ProjectBatchF32To(dst, x, batch); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("dequant_x4_candidate", func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					if err := tc.alt(m, dst, x, batch); err != nil {
						b.Fatal(err)
					}
				}
			})
		})
	}
}

func syntheticProjectBenchMatrix(t testing.TB, qtype QuantType, inDim, outDim int) *QuantMatrix {
	t.Helper()
	m := &QuantMatrix{Name: "synthetic.bench", QType: qtype, InDim: inDim, OutDim: outDim}
	rowBytes, err := m.RowBytes()
	if err != nil {
		t.Fatal(err)
	}
	m.Raw = make([]byte, rowBytes*outDim)
	for r := 0; r < outDim; r++ {
		row := m.Raw[r*rowBytes : (r+1)*rowBytes]
		switch qtype {
		case QuantQ5_0:
			for b := 0; b < inDim/32; b++ {
				blk := row[b*22 : (b+1)*22]
				binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(0.05+float32(r+b)*0.01))
				binary.LittleEndian.PutUint32(blk[2:6], uint32(0xa5a50000|r<<3|b))
				for i := 0; i < 16; i++ {
					blk[6+i] = byte((r*13 + b*17 + i*7) & 0xff)
				}
			}
		case QuantQ8_0:
			for b := 0; b < inDim/32; b++ {
				blk := row[b*34 : (b+1)*34]
				binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(0.07+float32(r+b)*0.013))
				for i := 0; i < 32; i++ {
					blk[2+i] = byte(int8((r*3+b*5+i)%15 - 7))
				}
			}
		case QuantQ6_K:
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
