package nvidia

import (
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
)

const cudaGraphLiveEnv = "GO_PHERENCE_RUN_CUDA_GRAPH_LIVE"

func TestCapturedGraphSegmentValidation(t *testing.T) {
	if _, err := newCapturedGraphSegment("", &CapturedGraph{}); err == nil {
		t.Fatal("expected empty shape key error")
	}
	if _, err := newCapturedGraphSegment("batch=1:hidden=16", nil); err == nil {
		t.Fatal("expected nil graph error")
	}
}

func TestCapturedGraphSegmentShapeKeyGuard(t *testing.T) {
	seg, err := newCapturedGraphSegment("batch=1:hidden=16", &CapturedGraph{})
	if err != nil {
		t.Fatalf("newCapturedGraphSegment: %v", err)
	}
	if err := seg.Launch("batch=1:hidden=32"); err == nil || !strings.Contains(err.Error(), "shape mismatch") {
		t.Fatalf("shape guard error = %v, want mismatch", err)
	}
	if err := seg.Launch("batch=1:hidden=16"); err == nil || !strings.Contains(err.Error(), "nil CUDA graph executable") {
		t.Fatalf("matching shape should defer to wrapped graph launch error, got %v", err)
	}
}

func TestCapturedGraphSegmentDestroyIsIdempotent(t *testing.T) {
	seg, err := newCapturedGraphSegment("batch=1:hidden=16", &CapturedGraph{})
	if err != nil {
		t.Fatalf("newCapturedGraphSegment: %v", err)
	}
	seg.Destroy()
	seg.Destroy()
	if seg.graph != nil {
		t.Fatal("destroy should clear wrapped graph")
	}
}

func TestCUDAGraphBatch1SegmentLiveParity(t *testing.T) {
	requireCUDAGraphBatch1Live(t)

	const hidden = 1024
	eager := makeCUDAGraphBatch1Fixture(t, hidden)
	captured := makeCUDAGraphBatch1Fixture(t, hidden)

	if !eager.runGPUOnly() {
		t.Fatal("eager preflight fell back from GPU kernels")
	}
	if err := SyncErr(); err != nil {
		t.Fatalf("eager sync: %v", err)
	}
	want := append([]float32(nil), eager.out.Data()...)

	captured.prepareForCapture(t)
	seg := capturePreparedCUDAGraphBatch1Segment(t, captured)
	defer seg.Destroy()

	if err := seg.Launch(captured.shapeKey()); err != nil {
		t.Fatalf("captured launch: %v", err)
	}
	if err := SyncErr(); err != nil {
		t.Fatalf("captured sync: %v", err)
	}
	got := append([]float32(nil), captured.out.Data()...)

	maxDiff, maxIdx := maxAbsDiff(got, want)
	if maxDiff > 1e-6 {
		t.Fatalf("captured output mismatch: maxDiff=%g idx=%d got=%g want=%g", maxDiff, maxIdx, got[maxIdx], want[maxIdx])
	}
	t.Logf("captured batch-1 segment parity maxDiff=%g", maxDiff)
}

func BenchmarkCUDAGraphBatch1Segment(b *testing.B) {
	requireCUDAGraphBatch1Benchmark(b)

	const hidden = 1024
	eager := makeCUDAGraphBatch1Fixture(b, hidden)
	captured := makeCUDAGraphBatch1Fixture(b, hidden)
	captureOnly := makeCUDAGraphBatch1Fixture(b, hidden)

	captured.prepareForCapture(b)
	seg := capturePreparedCUDAGraphBatch1Segment(b, captured)
	b.Cleanup(seg.Destroy)
	if err := seg.Launch(captured.shapeKey()); err != nil {
		b.Fatalf("warm graph launch: %v", err)
	}
	if err := SyncErr(); err != nil {
		b.Fatalf("warm graph sync: %v", err)
	}

	captureOnly.prepareForCapture(b)

	b.Run("eager", func(b *testing.B) {
		if !eager.runGPUOnly() {
			b.Fatal("warm eager launch fell back from GPU kernels")
		}
		if err := SyncErr(); err != nil {
			b.Fatalf("warm eager sync: %v", err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if !eager.runGPUOnly() {
				b.Fatal("eager launch fell back from GPU kernels")
			}
			SyncForTiming()
		}
	})

	b.Run("captured_warm_launch", func(b *testing.B) {
		if err := seg.Launch(captured.shapeKey()); err != nil {
			b.Fatalf("warm captured launch: %v", err)
		}
		if err := SyncErr(); err != nil {
			b.Fatalf("warm captured sync: %v", err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := seg.Launch(captured.shapeKey()); err != nil {
				b.Fatal(err)
			}
			SyncForTiming()
		}
	})

	b.Run("capture_instantiate", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			seg := capturePreparedCUDAGraphBatch1Segment(b, captureOnly)
			seg.Destroy()
		}
	})
}

type cudaGraphBatch1Fixture struct {
	hidden   int
	eps      float32
	scale    float32
	hiddenIn *DevBuf
	residual *DevBuf
	weight   *DevBuf
	normed   *DevBuf
	scaled   *DevBuf
	out      *DevBuf
}

func requireCUDAGraphBatch1Benchmark(tb testing.TB) {
	tb.Helper()
	if !GraphsReady() || !SgemmReady() || fnRmsNorm == 0 || fnVecScale == 0 || fnVecAdd == 0 {
		tb.Skip("CUDA graph benchmark kernels not available")
	}
}

func requireCUDAGraphBatch1Live(tb testing.TB) {
	tb.Helper()
	requireCUDAGraphBatch1Benchmark(tb)
	if _, ok := tb.(*testing.T); ok && os.Getenv(cudaGraphLiveEnv) != "1" {
		tb.Skipf("set %s=1 to run the live CUDA graph batch-1 segment parity test", cudaGraphLiveEnv)
	}
}

func makeCUDAGraphBatch1Fixture(tb testing.TB, hidden int) *cudaGraphBatch1Fixture {
	tb.Helper()
	hiddenIn, residual, weight := makeCUDAGraphBatch1Data(hidden)
	fx := &cudaGraphBatch1Fixture{
		hidden:   hidden,
		eps:      1e-5,
		scale:    0.75,
		hiddenIn: NewDevBufFrom(hiddenIn),
		residual: NewDevBufFrom(residual),
		weight:   NewDevBufFrom(weight),
	}
	for name, buf := range map[string]*DevBuf{
		"hidden":   fx.hiddenIn,
		"residual": fx.residual,
		"weight":   fx.weight,
	} {
		if err := buf.ToGPU(); err != nil {
			tb.Fatalf("%s ToGPU: %v", name, err)
		}
		tb.Cleanup(buf.Free)
	}
	var err error
	if fx.normed, err = NewDevBufGPU(hidden); err != nil {
		tb.Fatalf("normed alloc: %v", err)
	}
	if fx.scaled, err = NewDevBufGPU(hidden); err != nil {
		tb.Fatalf("scaled alloc: %v", err)
	}
	if fx.out, err = NewDevBufGPU(hidden); err != nil {
		tb.Fatalf("out alloc: %v", err)
	}
	for _, buf := range []*DevBuf{fx.normed, fx.scaled, fx.out} {
		tb.Cleanup(buf.Free)
	}
	fx.zeroOutputs(tb)
	return fx
}

func makeCUDAGraphBatch1Data(hidden int) ([]float32, []float32, []float32) {
	hiddenIn := make([]float32, hidden)
	residual := make([]float32, hidden)
	weight := make([]float32, hidden)
	for i := 0; i < hidden; i++ {
		hiddenIn[i] = float32((i%23)-11) * 0.0625
		if i&1 == 1 {
			hiddenIn[i] = -hiddenIn[i]
		}
		residual[i] = float32((i%19)-9) * 0.03125
		if (i/3)&1 == 1 {
			residual[i] = -residual[i]
		}
		weight[i] = 0.5 + float32((i%29)+1)*0.015625
	}
	return hiddenIn, residual, weight
}

func (fx *cudaGraphBatch1Fixture) shapeKey() string {
	return fmt.Sprintf("batch=1:hidden=%d:ops=rmsnorm-scale-add", fx.hidden)
}

func (fx *cudaGraphBatch1Fixture) runGPUOnly() bool {
	ok := DevRMSNormOK(fx.normed, fx.hiddenIn, fx.weight, fx.eps)
	DevScale(fx.scaled, fx.normed, fx.scale)
	DevAdd(fx.out, fx.residual, fx.scaled)
	return ok && fx.normed.OnGPU() && fx.scaled.OnGPU() && fx.out.OnGPU()
}

func (fx *cudaGraphBatch1Fixture) prepareForCapture(tb testing.TB) {
	tb.Helper()
	if !fx.runGPUOnly() {
		tb.Fatal("capture preflight fell back from GPU kernels")
	}
	if err := SyncErr(); err != nil {
		tb.Fatalf("capture preflight sync: %v", err)
	}
	fx.zeroOutputs(tb)
}

func (fx *cudaGraphBatch1Fixture) zeroOutputs(tb testing.TB) {
	tb.Helper()
	for name, buf := range map[string]*DevBuf{
		"normed": fx.normed,
		"scaled": fx.scaled,
		"out":    fx.out,
	} {
		if err := ZeroFloat32Buffer(buf.GPUBuffer(), fx.hidden); err != nil {
			tb.Fatalf("zero %s: %v", name, err)
		}
		buf.MarkOnGPU()
	}
}

func capturePreparedCUDAGraphBatch1Segment(tb testing.TB, fx *cudaGraphBatch1Fixture) *capturedGraphSegment {
	tb.Helper()
	if err := BeginCapture(); err != nil {
		tb.Fatalf("begin capture: %v", err)
	}
	gpuOK := fx.runGPUOnly()
	cg, err := EndCapture()
	if err != nil {
		tb.Fatalf("end capture: %v", err)
	}
	if !gpuOK {
		cg.Destroy()
		tb.Fatal("capture path fell back from GPU kernels")
	}
	seg, err := newCapturedGraphSegment(fx.shapeKey(), cg)
	if err != nil {
		cg.Destroy()
		tb.Fatalf("wrap captured graph: %v", err)
	}
	return seg
}

func maxAbsDiff(got, want []float32) (float64, int) {
	if len(got) != len(want) {
		return math.Inf(1), -1
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
	return maxDiff, maxIdx
}
