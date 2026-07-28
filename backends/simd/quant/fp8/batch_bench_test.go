package fp8

import (
	"fmt"
	"math"
	"testing"
)

var fp8FiniteCodes = [...]byte{
	0x00, 0x01, 0x05, 0x10, 0x1d, 0x25, 0x30, 0x38,
	0x39, 0x41, 0x4c, 0x56, 0x62, 0x6a, 0x76, 0x7e,
	0x81, 0x85, 0x90, 0x9d, 0xa8, 0xb8, 0xc4, 0xd2,
	0xe8, 0xfe,
}

func makeBenchmarkLinear(outDim, inDim int) Linear {
	weight := make([]byte, outDim*inDim)
	scale := make([]float32, outDim)
	bias := make([]float32, outDim)
	for row := 0; row < outDim; row++ {
		scale[row] = float32((row%7)+1) * (1.0 / 256.0)
		bias[row] = float32((row%19)-9) * (1.0 / 64.0)
		wRow := weight[row*inDim : (row+1)*inDim]
		for col := 0; col < inDim; col++ {
			wRow[col] = fp8FiniteCodes[(row*7+col*11)%len(fp8FiniteCodes)]
		}
	}
	return Linear{OutDim: outDim, InDim: inDim, Weight: weight, Scale: scale, Bias: bias}
}

func makeBenchmarkBatchInput(batch, inDim int) []float32 {
	x := make([]float32, batch*inDim)
	for b := 0; b < batch; b++ {
		row := x[b*inDim : (b+1)*inDim]
		for i := 0; i < inDim; i++ {
			v := float32(((b+3)*(i%29+1))%41-20) * 0.0625
			if (b+i)&1 != 0 {
				v = -v
			}
			row[i] = v
		}
	}
	return x
}

func repeatedGemvTo(l Linear, x []float32, batch int) ([]float32, error) {
	out := make([]float32, batch*l.OutDim)
	for b := 0; b < batch; b++ {
		if err := l.GemvTo(x[b*l.InDim:(b+1)*l.InDim], out[b*l.OutDim:(b+1)*l.OutDim]); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func repeatedGemvToDynamicToken(l Linear, x []float32, batch int) ([]float32, error) {
	out := make([]float32, batch*l.OutDim)
	scratch := make([]float32, l.InDim)
	for b := 0; b < batch; b++ {
		if err := l.GemvToDynamicToken(x[b*l.InDim:(b+1)*l.InDim], out[b*l.OutDim:(b+1)*l.OutDim], scratch); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func fp8BatchTolerance(inDim int) float64 {
	return 1e-4 + 2e-6*float64(inDim)
}

func requireCloseBatchOutput(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d len(want)=%d", len(got), len(want))
	}
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			t.Fatalf("out[%d]=%g want=%g diff=%g tol=%g", i, got[i], want[i], diff, tol)
		}
	}
}

func TestDecodeE4M3ArithmeticMatchesLUT(t *testing.T) {
	for code := 0; code < 256; code++ {
		got := decodeE4M3Arithmetic(byte(code))
		want := DecodeE4M3(byte(code))
		if math.IsNaN(float64(want)) {
			if !math.IsNaN(float64(got)) {
				t.Fatalf("decodeE4M3Arithmetic(0x%02x)=%g want NaN", code, got)
			}
			continue
		}
		if got != want {
			t.Fatalf("decodeE4M3Arithmetic(0x%02x)=%g want %g", code, got, want)
		}
	}
}

func TestBatchGemvToMatchesRepeatedGemv1536x2048(t *testing.T) {
	const (
		inDim  = 1536
		outDim = 2048
	)
	l := makeBenchmarkLinear(outDim, inDim)
	tol := fp8BatchTolerance(inDim)
	for _, batch := range []int{4, 8} {
		t.Run(fmt.Sprintf("batch_%d", batch), func(t *testing.T) {
			x := makeBenchmarkBatchInput(batch, inDim)
			want, err := repeatedGemvTo(l, x, batch)
			if err != nil {
				t.Fatal(err)
			}

			got := make([]float32, len(want))
			if err := l.BatchGemvTo(x, got, batch); err != nil {
				t.Fatal(err)
			}
			requireCloseBatchOutput(t, got, want, tol)

			gotBuf := make([]float32, len(want))
			if err := l.BatchGemvToBuf(x, gotBuf, batch, make([]float32, inDim)); err != nil {
				t.Fatal(err)
			}
			requireCloseBatchOutput(t, gotBuf, want, tol)
		})
	}
}

func TestBatchGemvDynamicTokenMatchesRepeatedGemv1536x2048(t *testing.T) {
	const (
		inDim  = 1536
		outDim = 2048
	)
	l := makeBenchmarkLinear(outDim, inDim)
	tol := fp8BatchTolerance(inDim)
	for _, batch := range []int{4, 8} {
		t.Run(fmt.Sprintf("batch_%d", batch), func(t *testing.T) {
			x := makeBenchmarkBatchInput(batch, inDim)
			want, err := repeatedGemvToDynamicToken(l, x, batch)
			if err != nil {
				t.Fatal(err)
			}
			got := make([]float32, len(want))
			if err := l.BatchGemvToBufDynamicToken(x, got, batch, make([]float32, inDim), make([]float32, batch*inDim)); err != nil {
				t.Fatal(err)
			}
			requireCloseBatchOutput(t, got, want, tol)
		})
	}
}

func TestBatchGemvToBufAllocs(t *testing.T) {
	l := makeBenchmarkLinear(384, 256)
	x := makeBenchmarkBatchInput(4, 256)
	out := make([]float32, 4*l.OutDim)
	wf32 := make([]float32, l.InDim)
	var err error
	allocs := testing.AllocsPerRun(100, func() {
		err = l.BatchGemvToBuf(x, out, 4, wf32)
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("BatchGemvToBuf allocs/run=%g want 0", allocs)
	}
}

func TestBatchGemvToBufDynamicTokenAllocs(t *testing.T) {
	l := makeBenchmarkLinear(384, 256)
	x := makeBenchmarkBatchInput(4, 256)
	out := make([]float32, 4*l.OutDim)
	wf32 := make([]float32, l.InDim)
	xq := make([]float32, 4*l.InDim)
	var err error
	allocs := testing.AllocsPerRun(100, func() {
		err = l.BatchGemvToBufDynamicToken(x, out, 4, wf32, xq)
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("BatchGemvToBufDynamicToken allocs/run=%g want 0", allocs)
	}
}

func benchmarkBatchGemv1536x2048(b *testing.B, batch int) {
	const (
		inDim  = 1536
		outDim = 2048
	)
	l := makeBenchmarkLinear(outDim, inDim)
	x := makeBenchmarkBatchInput(batch, inDim)
	bytes := int64(batch * inDim * outDim * 5)

	b.Run("batch_gemv", func(b *testing.B) {
		out := make([]float32, batch*outDim)
		b.ReportAllocs()
		b.SetBytes(bytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := l.BatchGemvTo(x, out, batch); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("batch_gemv_buf", func(b *testing.B) {
		out := make([]float32, batch*outDim)
		wf32 := make([]float32, inDim)
		b.ReportAllocs()
		b.SetBytes(bytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := l.BatchGemvToBuf(x, out, batch, wf32); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("repeated_gemv", func(b *testing.B) {
		out := make([]float32, batch*outDim)
		b.ReportAllocs()
		b.SetBytes(bytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for row := 0; row < batch; row++ {
				if err := l.GemvTo(x[row*inDim:(row+1)*inDim], out[row*outDim:(row+1)*outDim]); err != nil {
					b.Fatal(err)
				}
			}
		}
	})
}

func BenchmarkFP8BatchGemvBatch4_1536x2048(b *testing.B) { benchmarkBatchGemv1536x2048(b, 4) }
func BenchmarkFP8BatchGemvBatch8_1536x2048(b *testing.B) { benchmarkBatchGemv1536x2048(b, 8) }

func benchmarkBatchGemvDynamicToken1536x2048(b *testing.B, batch int) {
	const (
		inDim  = 1536
		outDim = 2048
	)
	l := makeBenchmarkLinear(outDim, inDim)
	x := makeBenchmarkBatchInput(batch, inDim)
	bytes := int64(batch * inDim * outDim * 5)

	b.Run("dynamic_token_batch_gemv_buf", func(b *testing.B) {
		out := make([]float32, batch*outDim)
		wf32 := make([]float32, inDim)
		xq := make([]float32, batch*inDim)
		b.ReportAllocs()
		b.SetBytes(bytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := l.BatchGemvToBufDynamicToken(x, out, batch, wf32, xq); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("repeated_dynamic_token_gemv", func(b *testing.B) {
		out := make([]float32, batch*outDim)
		scratch := make([]float32, inDim)
		b.ReportAllocs()
		b.SetBytes(bytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for row := 0; row < batch; row++ {
				if err := l.GemvToDynamicToken(x[row*inDim:(row+1)*inDim], out[row*outDim:(row+1)*outDim], scratch); err != nil {
					b.Fatal(err)
				}
			}
		}
	})
}

func BenchmarkFP8BatchGemvDynamicTokenBatch4_1536x2048(b *testing.B) {
	benchmarkBatchGemvDynamicToken1536x2048(b, 4)
}

func BenchmarkFP8BatchGemvDynamicTokenBatch8_1536x2048(b *testing.B) {
	benchmarkBatchGemvDynamicToken1536x2048(b, 8)
}

func BenchmarkDotE4M3DecodeVariants(b *testing.B) {
	for _, cols := range []int{1536, 2816} {
		x := makeBenchmarkBatchInput(1, cols)
		w := make([]byte, cols)
		for i := range w {
			w[i] = fp8FiniteCodes[(i*13)%len(fp8FiniteCodes)]
		}
		b.Run(fmt.Sprintf("cols_%d/lut_dispatch", cols), func(b *testing.B) {
			var sink float32
			b.ReportAllocs()
			b.SetBytes(int64(cols * 5))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sink += dotE4M3(x[:cols], w)
			}
			_ = sink
		})
		b.Run(fmt.Sprintf("cols_%d/lut_scalar", cols), func(b *testing.B) {
			var sink float32
			b.ReportAllocs()
			b.SetBytes(int64(cols * 5))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sink += dotE4M3Scalar(x[:cols], w)
			}
			_ = sink
		})
		b.Run(fmt.Sprintf("cols_%d/arithmetic_scalar", cols), func(b *testing.B) {
			var sink float32
			b.ReportAllocs()
			b.SetBytes(int64(cols * 5))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sink += dotE4M3ArithmeticScalar(x[:cols], w)
			}
			_ = sink
		})
	}
}
