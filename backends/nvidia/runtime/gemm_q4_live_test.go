package nvidia

import (
	"fmt"
	"math"
	"testing"

	simdq4 "github.com/rcarmo/go-pherence/backends/simd/quant/q4"
)

func packBenchmarkQ4Sym(a0, a1, a2, a3, a4, a5, a6, a7 int32) int32 {
	return (a0 & 0xF) |
		((a1 & 0xF) << 4) |
		((a2 & 0xF) << 8) |
		((a3 & 0xF) << 12) |
		((a4 & 0xF) << 16) |
		((a5 & 0xF) << 20) |
		((a6 & 0xF) << 24) |
		((a7 & 0xF) << 28)
}

func makeLiveGemmQ4Inputs(batch, inDim, outDim int) ([]float32, []int32, []int32, []float32) {
	x := make([]float32, batch*inDim)
	for b := 0; b < batch; b++ {
		row := x[b*inDim : (b+1)*inDim]
		for i := 0; i < inDim; i++ {
			v := float32(((b+1)*(i%31+1))%37-18) * 0.0625
			if (b+i)&1 == 1 {
				v = -v
			}
			row[i] = v
		}
	}
	qweight := make([]int32, (inDim/8)*outDim)
	for pack := 0; pack < inDim/8; pack++ {
		for col := 0; col < outDim; col++ {
			qweight[pack*outDim+col] = packBenchmarkQ4Sym(
				int32((pack+col+0)&0xF),
				int32((pack+col+1)&0xF),
				int32((pack+col+2)&0xF),
				int32((pack+col+3)&0xF),
				int32((pack+col+4)&0xF),
				int32((pack+col+5)&0xF),
				int32((pack+col+6)&0xF),
				int32((pack+col+7)&0xF),
			)
		}
	}
	const groupSize = 128
	groups := inDim / groupSize
	gIdx := make([]int32, inDim)
	for i := range gIdx {
		gIdx[i] = int32(i / groupSize)
	}
	scales := make([]float32, groups*outDim)
	for g := 0; g < groups; g++ {
		row := scales[g*outDim : (g+1)*outDim]
		for col := 0; col < outDim; col++ {
			v := float32(((g+3)*(col%17+1))%29-14) * 0.0625
			if v == 0 {
				v = 0.125
			}
			row[col] = v
		}
	}
	return x, qweight, gIdx, scales
}

func requireLiveGemmQ4GPU(tb testing.TB) {
	tb.Helper()
	if !SgemmReady() || !Q4Ready() || !BatchGEMMReady() {
		tb.Skip("no GPU or Q4/batch kernels")
	}
}

func requireCloseBatchQ4Output(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d len(want)=%d", len(got), len(want))
	}
	maxDiff := 0.0
	maxIdx := 0
	for i := range got {
		diff := math.Abs(float64(got[i] - want[i]))
		if diff > maxDiff {
			maxDiff = diff
			maxIdx = i
		}
	}
	if maxDiff > tol {
		t.Fatalf("maxDiff=%g at %d: got=%g want=%g tol=%g", maxDiff, maxIdx, got[maxIdx], want[maxIdx], tol)
	}
	t.Logf("maxDiff=%g tol=%g", maxDiff, tol)
}

func runLiveGemmQ4Parity(t *testing.T, batch, inDim, outDim int) {
	t.Helper()
	requireLiveGemmQ4GPU(t)

	x, qweight, gIdx, scales := makeLiveGemmQ4Inputs(batch, inDim, outDim)
	want := make([]float32, batch*outDim)
	simdq4.GemmSym(want, x, batch, qweight, gIdx, scales, inDim, outDim)

	w, err := UploadQuantWeight(qweight, gIdx, scales, inDim, outDim)
	if err != nil {
		t.Fatalf("upload: %v", err)
	}
	defer w.Free()

	inp := NewDevBuf(batch * inDim)
	copy(inp.Data(), x)
	inp.MarkDirty()
	defer inp.Free()

	out := NewDevBuf(batch * outDim)
	defer out.Free()

	GemmQ4(out, inp, w, batch)
	if err := SyncErr(); err != nil {
		t.Fatalf("sync: %v", err)
	}

	got := append([]float32(nil), out.Data()...)
	tol := 0.02
	if inDim >= 1024 {
		tol = 0.05
	}
	requireCloseBatchQ4Output(t, got, want, tol)
}

func TestGemmQ4LiveParity1536x2048(t *testing.T) {
	const (
		inDim  = 1536
		outDim = 2048
	)
	for _, batch := range []int{4, 8} {
		t.Run(fmt.Sprintf("batch_%d", batch), func(t *testing.T) {
			runLiveGemmQ4Parity(t, batch, inDim, outDim)
		})
	}
}

func TestGemmQ4LiveTailParity1536x2050(t *testing.T) {
	const (
		inDim  = 1536
		outDim = 2050
	)
	for _, batch := range []int{4, 8} {
		t.Run(fmt.Sprintf("batch_%d", batch), func(t *testing.T) {
			runLiveGemmQ4Parity(t, batch, inDim, outDim)
		})
	}
}

func benchmarkGemmQ4GPU(b *testing.B, batch int) {
	const (
		inDim  = 1536
		outDim = 2048
	)
	requireLiveGemmQ4GPU(b)

	x, qweight, gIdx, scales := makeLiveGemmQ4Inputs(batch, inDim, outDim)
	w, err := UploadQuantWeight(qweight, gIdx, scales, inDim, outDim)
	if err != nil {
		b.Fatalf("upload: %v", err)
	}
	b.Cleanup(w.Free)

	inp := NewDevBuf(batch * inDim)
	copy(inp.Data(), x)
	inp.MarkDirty()
	if err := inp.ToGPU(); err != nil {
		b.Fatalf("input ToGPU: %v", err)
	}
	b.Cleanup(inp.Free)

	out := NewDevBuf(batch * outDim)
	if err := out.ToGPU(); err != nil {
		b.Fatalf("output ToGPU: %v", err)
	}
	b.Cleanup(out.Free)

	rows := make([]*DevBuf, batch)
	outs := make([]*DevBuf, batch)
	for i := 0; i < batch; i++ {
		rows[i] = inp.Slice(i*inDim, inDim)
		outs[i] = out.Slice(i*outDim, outDim)
	}

	bytes := int64(batch * inDim * outDim * 4)

	b.Run("batched_gemm", func(b *testing.B) {
		GemmQ4(out, inp, w, batch)
		if err := SyncErr(); err != nil {
			b.Fatalf("warmup sync: %v", err)
		}
		b.ReportAllocs()
		b.SetBytes(bytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			GemmQ4(out, inp, w, batch)
			SyncForTiming()
		}
	})

	b.Run("repeated_gemv", func(b *testing.B) {
		for i := 0; i < batch; i++ {
			GemvQ4(outs[i], rows[i], w)
		}
		if err := SyncErr(); err != nil {
			b.Fatalf("warmup sync: %v", err)
		}
		b.ReportAllocs()
		b.SetBytes(bytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for row := 0; row < batch; row++ {
				GemvQ4(outs[row], rows[row], w)
			}
			SyncForTiming()
		}
	})
}

func BenchmarkGemmQ4Batch4_1536x2048(b *testing.B) { benchmarkGemmQ4GPU(b, 4) }
func BenchmarkGemmQ4Batch8_1536x2048(b *testing.B) { benchmarkGemmQ4GPU(b, 8) }
