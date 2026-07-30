package nvidia

import (
	"fmt"
	"math"
	"os"
	"testing"
)

func TestMLXSelectedExpertPersistentLiveParity(t *testing.T) {
	requireMLXSelectedExpertPersistentLive(t)
	const (
		experts   = 4
		hidden    = 512
		outDim    = 1024
		groupSize = 64
	)
	fx := makeMLXSelectedExpertPersistentFixture(t, experts, hidden, outDim, groupSize)
	for _, tc := range []struct {
		name string
		work []uint32
	}{
		{name: "work4_nonzero_order", work: []uint32{3, 1, 2, 0}},
		{name: "work3_tail", work: []uint32{2, 3, 1}},
		{name: "work5_tail_repeat", work: []uint32{3, 1, 3, 2, 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			runMLXSelectedExpertPersistentParity(t, fx, tc.work)
		})
	}
}

func BenchmarkMLXSelectedExpertPersistentCandidates(b *testing.B) {
	if !SgemmReady() || fnMLXGemv == 0 || fnMLXSelectedExpertPersistent == 0 {
		b.Skip("MLX selected expert kernels not available")
	}
	const (
		experts   = 4
		hidden    = 512
		outDim    = 1024
		groupSize = 64
	)
	fx := makeMLXSelectedExpertPersistentFixture(b, experts, hidden, outDim, groupSize)
	for _, tc := range []struct {
		name string
		work []uint32
	}{
		{name: "work4_nonzero_order", work: []uint32{3, 1, 2, 0}},
		{name: "work3_tail", work: []uint32{2, 3, 1}},
		{name: "work5_tail_repeat", work: []uint32{3, 1, 3, 2, 0}},
	} {
		b.Run(tc.name, func(b *testing.B) {
			benchmarkMLXSelectedExpertPersistent(b, fx, tc.work)
		})
	}
}

func requireMLXSelectedExpertPersistentLive(tb testing.TB) {
	tb.Helper()
	if !SgemmReady() || fnMLXGemv == 0 || fnMLXSelectedExpertPersistent == 0 {
		tb.Skip("MLX selected expert kernels not available")
	}
	if _, ok := tb.(*testing.T); ok && os.Getenv("GO_PHERENCE_RUN_MLX_SELECTED_EXPERT_PERSISTENT") != "1" {
		tb.Skip("set GO_PHERENCE_RUN_MLX_SELECTED_EXPERT_PERSISTENT=1 to run the MLX selected expert persistent live parity test")
	}
}

type mlxSelectedExpertPersistentFixture struct {
	x       *DevBuf
	weights []*GPUMLXWeight
	inDim   int
	outDim  int
}

func makeMLXSelectedExpertPersistentFixture(tb testing.TB, experts, inDim, outDim, groupSize int) mlxSelectedExpertPersistentFixture {
	tb.Helper()
	xData := make([]float32, inDim)
	for i := range xData {
		v := float32((i%29)-14) * 0.0625
		if (i/3)&1 == 1 {
			v = -v
		}
		xData[i] = v
	}
	xBuf := NewDevBufFrom(xData)
	if err := xBuf.ToGPU(); err != nil {
		tb.Fatalf("input ToGPU: %v", err)
	}
	tb.Cleanup(xBuf.Free)

	weights := make([]*GPUMLXWeight, experts)
	for expert := 0; expert < experts; expert++ {
		w, scales, biases := makeMLXSelectedExpertWeight(expert, inDim, outDim, groupSize)
		gpu, err := UploadMLXWeightNative(w, scales, biases, inDim, outDim, groupSize)
		if err != nil {
			tb.Fatalf("upload expert %d: %v", expert, err)
		}
		weights[expert] = gpu
	}
	tb.Cleanup(func() {
		for _, w := range weights {
			if w != nil {
				w.Free()
			}
		}
	})

	return mlxSelectedExpertPersistentFixture{
		x:       xBuf,
		weights: weights,
		inDim:   inDim,
		outDim:  outDim,
	}
}

func makeMLXSelectedExpertWeight(expert, inDim, outDim, groupSize int) ([]uint32, []float32, []float32) {
	packedPerRow := inDim / 8
	groups := inDim / groupSize
	weight := make([]uint32, outDim*packedPerRow)
	scales := make([]float32, outDim*groups)
	biases := make([]float32, outDim*groups)
	for row := 0; row < outDim; row++ {
		for pack := 0; pack < packedPerRow; pack++ {
			var packed uint32
			for i := 0; i < 8; i++ {
				val := uint32((expert*13 + row*7 + pack*5 + i*3) & 0xF)
				packed |= val << (uint(i) * 4)
			}
			weight[row*packedPerRow+pack] = packed
		}
		for g := 0; g < groups; g++ {
			idx := row*groups + g
			scales[idx] = 0.015625 * float32(((expert+1)*(g+3)+row%11)%17+1)
			biases[idx] = 0.03125*float32((expert+g+row)%9) - 0.125
		}
	}
	return weight, scales, biases
}

func runMLXSelectedExpertPersistentParity(t *testing.T, fx mlxSelectedExpertPersistentFixture, work []uint32) {
	t.Helper()
	candidate := NewMLXSelectedExpertPersistentCandidate()
	defer candidate.Free()

	candidateOut, err := NewDevBufGPU(len(work) * fx.outDim)
	if err != nil {
		t.Fatalf("candidate out alloc: %v", err)
	}
	defer candidateOut.Free()
	if err := candidate.Run(candidateOut, fx.x, fx.weights, work); err != nil {
		t.Fatalf("candidate run: %v", err)
	}
	if err := SyncErr(); err != nil {
		t.Fatalf("candidate sync: %v", err)
	}
	got := append([]float32(nil), candidateOut.Data()...)

	wantOut, err := NewDevBufGPU(len(work) * fx.outDim)
	if err != nil {
		t.Fatalf("reference out alloc: %v", err)
	}
	defer wantOut.Free()
	for i, expert := range work {
		GemvMLXDirect(wantOut.Slice(i*fx.outDim, fx.outDim), fx.x, fx.weights[int(expert)])
	}
	if err := SyncErr(); err != nil {
		t.Fatalf("reference sync: %v", err)
	}
	want := append([]float32(nil), wantOut.Data()...)
	assertMLXSelectedExpertClose(t, got, want, 1e-5)
}

func assertMLXSelectedExpertClose(t *testing.T, got, want []float32, tol float64) {
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
		t.Fatalf("maxDiff=%g at %d got=%g want=%g tol=%g", maxDiff, maxIdx, got[maxIdx], want[maxIdx], tol)
	}
	t.Logf("maxDiff=%g tol=%g", maxDiff, tol)
}

func benchmarkMLXSelectedExpertPersistent(b *testing.B, fx mlxSelectedExpertPersistentFixture, work []uint32) {
	candidate := NewMLXSelectedExpertPersistentCandidate()
	b.Cleanup(candidate.Free)

	candidateOut, err := NewDevBufGPU(len(work) * fx.outDim)
	if err != nil {
		b.Fatalf("candidate out alloc: %v", err)
	}
	b.Cleanup(candidateOut.Free)
	repeatedOut, err := NewDevBufGPU(len(work) * fx.outDim)
	if err != nil {
		b.Fatalf("reference out alloc: %v", err)
	}
	b.Cleanup(repeatedOut.Free)

	bytes := int64(len(work) * fx.inDim * fx.outDim * 4)
	b.Run("persistent_candidate", func(b *testing.B) {
		if err := candidate.Run(candidateOut, fx.x, fx.weights, work); err != nil {
			b.Fatalf("warmup candidate: %v", err)
		}
		if err := SyncErr(); err != nil {
			b.Fatalf("warmup candidate sync: %v", err)
		}
		b.ReportAllocs()
		b.SetBytes(bytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := candidate.Run(candidateOut, fx.x, fx.weights, work); err != nil {
				b.Fatal(err)
			}
			SyncForTiming()
		}
	})

	b.Run("repeated_gemv_mlx_direct", func(b *testing.B) {
		for i, expert := range work {
			GemvMLXDirect(repeatedOut.Slice(i*fx.outDim, fx.outDim), fx.x, fx.weights[int(expert)])
		}
		if err := SyncErr(); err != nil {
			b.Fatalf("warmup repeated sync: %v", err)
		}
		b.ReportAllocs()
		b.SetBytes(bytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for row, expert := range work {
				GemvMLXDirect(repeatedOut.Slice(row*fx.outDim, fx.outDim), fx.x, fx.weights[int(expert)])
			}
			SyncForTiming()
		}
	})
}

func TestMLXSelectedExpertPersistentValidation(t *testing.T) {
	cand := NewMLXSelectedExpertPersistentCandidate()
	defer cand.Free()
	if err := cand.Run(nil, nil, nil, []uint32{0}); err == nil {
		t.Fatal("expected nil candidate input validation error")
	}
	if _, _, _, _, err := validateMLXSelectedExpertNativeWeights(nil); err == nil {
		t.Fatal("expected empty weight set error")
	}
	bad := &GPUMLXWeight{InDim: 16, OutDim: 4, Groups: 2, GroupSz: 8, QWeight: &Buffer{Ptr: 1, Size: 8}, Scales: &Buffer{Ptr: 1, Size: 32}, Biases: &Buffer{Ptr: 1, Size: 32}}
	if err := validateNativeGPUMLXWeight(bad); err == nil {
		t.Fatal("expected short native qweight error")
	}
	bad.GroupSz = 10
	if err := validateNativeGPUMLXWeight(bad); err == nil {
		t.Fatal("expected non-pack-aligned group size error")
	}
}

func ExampleMLXSelectedExpertPersistentCandidate() {
	fmt.Println("set GO_PHERENCE_RUN_MLX_SELECTED_EXPERT_PERSISTENT=1 and run go test ./backends/nvidia/runtime -run TestMLXSelectedExpertPersistentLiveParity -count=1")
	// Output:
	// set GO_PHERENCE_RUN_MLX_SELECTED_EXPERT_PERSISTENT=1 and run go test ./backends/nvidia/runtime -run TestMLXSelectedExpertPersistentLiveParity -count=1
}
