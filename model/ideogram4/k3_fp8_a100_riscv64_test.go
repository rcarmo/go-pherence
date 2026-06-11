//go:build riscv64

package ideogram4

import (
	"math"
	"os"
	"testing"
)

func syntheticFP8Linear(t testing.TB, inDim, outDim int) *FP8Linear {
	t.Helper()
	w := make([]byte, inDim*outDim)
	scale := make([]float32, outDim)
	bias := make([]float32, outDim)
	vals := []byte{0x00, 0x20, 0x28, 0x30, 0x34, 0x38, 0xb0, 0xb4, 0xb8, 0x3c, 0xbc}
	for i := range w {
		w[i] = vals[(i*7+i/3)%len(vals)]
	}
	for i := range scale {
		scale[i] = 0.015625 * (1 + float32(i%5)*0.25)
		bias[i] = float32((i%7)-3) * 0.01
	}
	spec := LinearSpec{Prefix: "test.a100", Role: RoleMLPW1, InDim: inDim, OutDim: outDim, Weight: "w", WeightScale: "s"}
	lin, err := NewFP8Linear(spec, w, scale, bias)
	if err != nil {
		t.Fatal(err)
	}
	return lin
}

func syntheticRows(batch, inDim int) []float32 {
	x := make([]float32, batch*inDim)
	for i := range x {
		x[i] = float32((i%23)-11) / 11
	}
	return x
}

func TestK3FP8ApplyBatchA100RowScale(t *testing.T) {
	if _, err := os.Stat("/proc/set_ai_thread"); err != nil {
		t.Skip("no A100 thread registration")
	}
	t.Setenv("GO_PHERENCE_IDEOGRAM4_K3", "1")
	t.Setenv("GO_PHERENCE_IDEOGRAM4_K3_A100_Q8", "1")
	t.Setenv("GO_PHERENCE_IDEOGRAM4_K3_A100_WORKERS", "2")
	t.Setenv("IME2_ACT_PACK_WORKERS", "2")

	const batch, inDim, outDim = 8, 64, 32
	lin := syntheticFP8Linear(t, inDim, outDim)
	x := syntheticRows(batch, inDim)
	want := make([]float32, batch*outDim)
	got := make([]float32, batch*outDim)
	for b := 0; b < batch; b++ {
		if err := lin.weight.GemvTo(x[b*inDim:(b+1)*inDim], want[b*outDim:(b+1)*outDim]); err != nil {
			t.Fatal(err)
		}
	}
	if ok, err := k3FP8Batch(lin, x, got, batch); !ok || err != nil {
		t.Fatalf("k3FP8Batch A100 ok=%v err=%v", ok, err)
	}
	var maxAbs, rmse float64
	for i := range want {
		d := float64(got[i] - want[i])
		ad := math.Abs(d)
		if ad > maxAbs {
			maxAbs = ad
		}
		rmse += d * d
	}
	rmse = math.Sqrt(rmse / float64(len(want)))
	if maxAbs > 0.20 || rmse > 0.05 {
		t.Fatalf("A100 row-scale mismatch maxAbs=%g rmse=%g", maxAbs, rmse)
	}
}

func BenchmarkK3FP8ApplyBatchCPU(b *testing.B) {
	const batch, inDim, outDim = 64, 512, 1024
	lin := syntheticFP8Linear(b, inDim, outDim)
	x := syntheticRows(batch, inDim)
	out := make([]float32, batch*outDim)
	b.SetBytes(int64(batch * inDim * outDim))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for r := 0; r < batch; r++ {
			if err := lin.weight.GemvTo(x[r*inDim:(r+1)*inDim], out[r*outDim:(r+1)*outDim]); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func BenchmarkK3FP8ApplyBatchRVVF16(b *testing.B) {
	const batch, inDim, outDim = 64, 512, 1024
	lin := syntheticFP8Linear(b, inDim, outDim)
	x := syntheticRows(batch, inDim)
	out := make([]float32, batch*outDim)
	b.Setenv("GO_PHERENCE_IDEOGRAM4_K3", "1")
	b.Setenv("GO_PHERENCE_IDEOGRAM4_K3_A100_Q8", "0")
	_ = lin.k3.ensureWeightF16(lin)
	b.SetBytes(int64(batch * inDim * outDim))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if ok, err := k3FP8Batch(lin, x, out, batch); !ok || err != nil {
			b.Fatalf("k3FP8Batch RVV ok=%v err=%v", ok, err)
		}
	}
}

func BenchmarkK3FP8ApplyBatchA100Q8(b *testing.B) {
	if _, err := os.Stat("/proc/set_ai_thread"); err != nil {
		b.Skip("no A100 thread registration")
	}
	const batch, inDim, outDim = 64, 512, 1024
	lin := syntheticFP8Linear(b, inDim, outDim)
	x := syntheticRows(batch, inDim)
	out := make([]float32, batch*outDim)
	b.Setenv("GO_PHERENCE_IDEOGRAM4_K3", "1")
	b.Setenv("GO_PHERENCE_IDEOGRAM4_K3_A100_Q8", "1")
	b.Setenv("GO_PHERENCE_IDEOGRAM4_K3_A100_WORKERS", "6")
	b.Setenv("IME2_ACT_PACK_WORKERS", "6")
	_ = lin.k3.ensureWeightQ80RowScale(lin)
	_ = k3A100WorkerPool()
	b.SetBytes(int64(batch * inDim * outDim))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if ok, err := k3FP8Batch(lin, x, out, batch); !ok || err != nil {
			b.Fatalf("k3FP8Batch A100 ok=%v err=%v", ok, err)
		}
	}
}

func benchmarkK3FP8ApplyBatchPath(b *testing.B, batch, inDim, outDim int, a100 bool) {
	if a100 {
		if _, err := os.Stat("/proc/set_ai_thread"); err != nil {
			b.Skip("no A100 thread registration")
		}
	}
	lin := syntheticFP8Linear(b, inDim, outDim)
	x := syntheticRows(batch, inDim)
	out := make([]float32, batch*outDim)
	b.Setenv("GO_PHERENCE_IDEOGRAM4_K3", "1")
	if a100 {
		b.Setenv("GO_PHERENCE_IDEOGRAM4_K3_A100_Q8", "1")
		b.Setenv("GO_PHERENCE_IDEOGRAM4_K3_A100_WORKERS", "6")
		b.Setenv("IME2_ACT_PACK_WORKERS", "6")
		_ = lin.k3.ensureWeightQ80RowScale(lin)
		_ = k3A100WorkerPool()
	} else {
		b.Setenv("GO_PHERENCE_IDEOGRAM4_K3_A100_Q8", "0")
		_ = lin.k3.ensureWeightF16(lin)
	}
	b.SetBytes(int64(batch * inDim * outDim))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if ok, err := k3FP8Batch(lin, x, out, batch); !ok || err != nil {
			b.Fatalf("k3FP8Batch ok=%v err=%v", ok, err)
		}
	}
}

func BenchmarkK3FP8ApplyBatchDiTShapeSquareRVVF16(b *testing.B) {
	benchmarkK3FP8ApplyBatchPath(b, 32, 4608, 4608, false)
}

func BenchmarkK3FP8ApplyBatchDiTShapeSquareA100Q8(b *testing.B) {
	benchmarkK3FP8ApplyBatchPath(b, 32, 4608, 4608, true)
}

func BenchmarkK3FP8ApplyBatchDiTShapeMLPUpRVVF16(b *testing.B) {
	benchmarkK3FP8ApplyBatchPath(b, 16, 4608, 12288, false)
}

func BenchmarkK3FP8ApplyBatchDiTShapeMLPUpA100Q8(b *testing.B) {
	benchmarkK3FP8ApplyBatchPath(b, 16, 4608, 12288, true)
}
