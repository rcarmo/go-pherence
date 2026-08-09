package gguf_test

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"
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
		{name: "q4_0_batch8", qtype: gguf.QuantQ4_0, inDim: 256, outDim: 11, batch: 8},
		{name: "q4_0_vnni_tail65", qtype: gguf.QuantQ4_0, inDim: 256, outDim: 11, batch: 65},
		{name: "q4_0_vnni_tail124", qtype: gguf.QuantQ4_0, inDim: 256, outDim: 11, batch: 124},
		{name: "q4_0_vnni_parallel_tail124", qtype: gguf.QuantQ4_0, inDim: 256, outDim: 513, batch: 124},
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
				if tc.qtype == gguf.QuantQ4_0 {
					if !gguf.GemvQ4_0Q8_0Rows(want, x[pos*tc.inDim:(pos+1)*tc.inDim], m) {
						t.Fatal("Q4_0 GEMV oracle failed")
					}
				} else if err := m.ProjectBatchF32To(want, x[pos*tc.inDim:(pos+1)*tc.inDim], 1); err != nil {
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

func TestQuantMatrixProjectBatchQ4_0ShapeDispatchRejectsUnretainedBatches(t *testing.T) {
	m := syntheticQuantMatrix(t, gguf.QuantQ4_0, 256, 513)
	for _, batch := range []int{1, 2, 4} {
		err := m.ProjectBatchF32To(make([]float32, batch*m.OutDim), make([]float32, batch*m.InDim), batch)
		if !errors.Is(err, gguf.ErrUnsupportedBatchProjection) {
			t.Fatalf("batch=%d error=%v, want ErrUnsupportedBatchProjection", batch, err)
		}
	}
}

func TestQuantMatrixProjectBatchQ4_0AcceptsOversizedSlices(t *testing.T) {
	const batch = 8
	m := syntheticQuantMatrix(t, gguf.QuantQ4_0, 256, 513)
	x := make([]float32, batch*m.InDim+17)
	dst := make([]float32, batch*m.OutDim+19)
	if err := m.ProjectBatchF32To(dst, x, batch); err != nil {
		t.Fatal(err)
	}
}

func TestQuantMatrixProjectBatchF32ToQ2KQ3KMatchesDequantRowOracle(t *testing.T) {
	cases := []struct {
		name   string
		qtype  gguf.QuantType
		inDim  int
		outDim int
		batch  int
	}{
		{name: "q2k_batch4", qtype: gguf.QuantQ2_K, inDim: 512, outDim: 13, batch: 4},
		{name: "q2k_tail5_parallel_rows", qtype: gguf.QuantQ2_K, inDim: 512, outDim: 1031, batch: 5},
		{name: "q3k_batch8", qtype: gguf.QuantQ3_K, inDim: 512, outDim: 11, batch: 8},
		{name: "q3k_tail6_parallel_rows", qtype: gguf.QuantQ3_K, inDim: 512, outDim: 1027, batch: 6},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := syntheticQuantMatrix(t, tc.qtype, tc.inDim, tc.outDim)
			x := make([]float32, tc.batch*tc.inDim)
			for i := range x {
				x[i] = float32((i%29)-14) * 0.01875
			}
			got := make([]float32, tc.batch*tc.outDim)
			if err := m.ProjectBatchF32To(got, x, tc.batch); err != nil {
				t.Fatal(err)
			}
			want := projectBatchDequantOracle(t, m, x, tc.batch)
			const tol = 1e-3
			for i := range want {
				if math.Abs(float64(got[i]-want[i])) > tol {
					t.Fatalf("idx=%d got=%g want=%g diff=%g", i, got[i], want[i], got[i]-want[i])
				}
			}
		})
	}
}

func TestQuantMatrixProjectBatchF32ToQ2KQ3KMalformed(t *testing.T) {
	cases := []struct {
		name string
		m    *gguf.QuantMatrix
		x    []float32
		want string
	}{
		{
			name: "q2k_in_dim_not_multiple_of_256",
			m:    &gguf.QuantMatrix{Name: "bad.q2k", QType: gguf.QuantQ2_K, Raw: make([]byte, 84), InDim: 255, OutDim: 1},
			x:    make([]float32, 255),
			want: "not multiple of 256",
		},
		{
			name: func() string { return "q2k_raw_short" }(),
			m: func() *gguf.QuantMatrix {
				m := syntheticQuantMatrix(t, gguf.QuantQ2_K, 256, 1)
				m.Raw = m.Raw[:len(m.Raw)-1]
				return m
			}(),
			x:    make([]float32, 256),
			want: "row 0 raw short",
		},
		{
			name: "q3k_in_dim_not_multiple_of_256",
			m:    &gguf.QuantMatrix{Name: "bad.q3k", QType: gguf.QuantQ3_K, Raw: make([]byte, 110), InDim: 255, OutDim: 1},
			x:    make([]float32, 255),
			want: "not multiple of 256",
		},
		{
			name: func() string { return "q3k_raw_short" }(),
			m: func() *gguf.QuantMatrix {
				m := syntheticQuantMatrix(t, gguf.QuantQ3_K, 256, 1)
				m.Raw = m.Raw[:len(m.Raw)-1]
				return m
			}(),
			x:    make([]float32, 256),
			want: "row 0 raw short",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dst := make([]float32, tc.m.OutDim)
			err := tc.m.ProjectBatchF32To(dst, tc.x, 1)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error=%v want substring %q", err, tc.want)
			}
		})
	}
}

func BenchmarkQuantMatrixProjectBatchF32ToQ2KQ3K(b *testing.B) {
	cases := []struct {
		name  string
		qtype gguf.QuantType
	}{
		{name: "q2k", qtype: gguf.QuantQ2_K},
		{name: "q3k", qtype: gguf.QuantQ3_K},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			const (
				inDim  = 2816
				outDim = 1024
				batch  = 8
			)
			m := syntheticQuantMatrix(b, tc.qtype, inDim, outDim)
			x := make([]float32, batch*inDim)
			for i := range x {
				x[i] = float32((i%31)-15) * 0.0125
			}
			dst := make([]float32, batch*outDim)
			b.Run("batch_path", func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					if err := m.ProjectBatchF32To(dst, x, batch); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("repeated_batch1", func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					for pos := 0; pos < batch; pos++ {
						if err := m.ProjectBatchF32To(dst[pos*outDim:(pos+1)*outDim], x[pos*inDim:(pos+1)*inDim], 1); err != nil {
							b.Fatal(err)
						}
					}
				}
			})
			b.Run("dequant_oracle", func(b *testing.B) {
				rowScratch := make([]float32, inDim)
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					projectBatchDequantOracleTo(b, dst, rowScratch, m, x, batch)
				}
			})
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
		case gguf.QuantQ4_0:
			for b := 0; b < inDim/32; b++ {
				blk := row[b*18 : (b+1)*18]
				binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(0.05+float32((r+b)%17)*0.003))
				for i := 0; i < 16; i++ {
					blk[2+i] = byte((r*13 + b*17 + i*7) & 0xff)
				}
			}
		case gguf.QuantQ2_K:
			for b := 0; b < inDim/256; b++ {
				blk := row[b*84 : (b+1)*84]
				for i := 0; i < 16; i++ {
					scale := byte(1 + (i+r+b)%15)
					minv := byte((i*3 + r + b) % 8)
					blk[i] = scale | (minv << 4)
				}
				for i := 0; i < 64; i++ {
					blk[16+i] = byte((i*7 + r*11 + b*13 + 3) & 0xff)
				}
				binary.LittleEndian.PutUint16(blk[80:82], half.F32ToF16(0.03125+float32(r+b)*0.001953125))
				binary.LittleEndian.PutUint16(blk[82:84], half.F32ToF16(0.0078125+float32((r+b)%5)*0.0009765625))
			}
		case gguf.QuantQ3_K:
			for b := 0; b < inDim/256; b++ {
				blk := row[b*110 : (b+1)*110]
				for i := 0; i < 32; i++ {
					blk[i] = byte((i*5 + r*7 + b*11 + 1) & 0xff)
				}
				for i := 0; i < 64; i++ {
					blk[32+i] = byte((i*9 + r*13 + b*3 + 5) & 0xff)
				}
				for i := 0; i < 12; i++ {
					blk[96+i] = byte((i*17 + r*19 + b*23 + 7) & 0xff)
				}
				binary.LittleEndian.PutUint16(blk[108:110], half.F32ToF16(0.0234375+float32(r+b)*0.0029296875))
			}
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
				for i := 0; i < 32; i++ {
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

func BenchmarkQuantMatrixProjectBatchQ4_0Gemma4Shapes(b *testing.B) {
	for _, outDim := range []int{512, 2048, 2560, 10240} {
		m := syntheticQuantMatrix(b, gguf.QuantQ4_0, 2560, outDim)
		for _, batch := range []int{8, 124} {
			x := make([]float32, batch*m.InDim)
			dst := make([]float32, batch*m.OutDim)
			b.Run(fmt.Sprintf("out%d/batch%d/batched", outDim, batch), func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					if err := m.ProjectBatchF32To(dst, x, batch); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run(fmt.Sprintf("out%d/batch%d/repeated", outDim, batch), func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					for pos := 0; pos < batch; pos++ {
						if !gguf.GemvQ4_0Q8_0Rows(dst[pos*m.OutDim:(pos+1)*m.OutDim], x[pos*m.InDim:(pos+1)*m.InDim], m) {
							b.Fatal("Q4_0 GEMV failed")
						}
					}
				}
			})
		}
	}
}

func projectBatchDequantOracle(t testing.TB, m *gguf.QuantMatrix, x []float32, batch int) []float32 {
	t.Helper()
	dst := make([]float32, batch*m.OutDim)
	projectBatchDequantOracleTo(t, dst, make([]float32, m.InDim), m, x, batch)
	return dst
}

func projectBatchDequantOracleTo(t testing.TB, dst, rowScratch []float32, m *gguf.QuantMatrix, x []float32, batch int) {
	t.Helper()
	for r := 0; r < m.OutDim; r++ {
		if err := m.DequantRowTo(rowScratch, r); err != nil {
			t.Fatal(err)
		}
		for pos := 0; pos < batch; pos++ {
			dst[pos*m.OutDim+r] = dotF32External(rowScratch[:m.InDim], x[pos*m.InDim:(pos+1)*m.InDim])
		}
	}
}

func dotF32External(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
