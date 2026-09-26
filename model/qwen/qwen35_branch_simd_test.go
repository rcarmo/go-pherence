package qwen

import (
	"fmt"
	"math"
	"runtime"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	cfg "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/tensor"
)

const simdProjectionTol = 1e-5

func TestSIMDBranchProjectionRowsResidues(t *testing.T) {
	const (
		in  = 19
		out = 17
	)
	oldSgemm := simd.HasSgemmAsm
	defer func() { simd.HasSgemmAsm = oldSgemm }()
	for _, dispatch := range []struct {
		name    string
		enabled bool
	}{{name: "host", enabled: oldSgemm}, {name: "disabled", enabled: false}} {
		simd.HasSgemmAsm = dispatch.enabled
		for rows := 1; rows <= 13; rows++ {
			for _, withScratch := range []bool{false, true} {
				name := fmt.Sprintf("%s/rows_%02d/scratch_%t", dispatch.name, rows, withScratch)
				t.Run(name, func(t *testing.T) {
					runSIMDBranchProjectionCase(t, rows, in, out, withScratch)
				})
			}
		}
	}
}

func TestSIMDBranchProjectionParallelTails(t *testing.T) {
	oldProcs := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(oldProcs)
	oldSgemm := simd.HasSgemmAsm
	defer func() { simd.HasSgemmAsm = oldSgemm }()
	for _, dispatch := range []struct {
		name    string
		enabled bool
	}{{name: "host", enabled: oldSgemm}, {name: "disabled", enabled: false}} {
		simd.HasSgemmAsm = dispatch.enabled
		for _, tc := range []struct {
			rows        int
			out         int
			withScratch bool
		}{{rows: 12, out: 511, withScratch: true}, {rows: 12, out: 512, withScratch: true}, {rows: 12, out: 513, withScratch: true}, {rows: 13, out: 511, withScratch: false}, {rows: 13, out: 513, withScratch: false}, {rows: 13, out: 511, withScratch: true}, {rows: 13, out: 513, withScratch: true}} {
			name := fmt.Sprintf("%s/rows_%02d/out_%03d/scratch_%t", dispatch.name, tc.rows, tc.out, tc.withScratch)
			t.Run(name, func(t *testing.T) {
				runSIMDBranchProjectionCase(t, tc.rows, 9, tc.out, tc.withScratch)
			})
		}
	}
}

func runSIMDBranchProjectionCase(t *testing.T, rows, in, out int, withScratch bool) {
	t.Helper()
	weights := make([]float32, out*in)
	for i := range weights {
		weights[i] = float32((i*7)%23-11) * 0.03125
	}
	weight := tensor.FromFloat32(weights, []int{out, in})
	packed, err := simd.PackSgemmNTWeights(weight.Data(), out, in, in)
	if err != nil {
		t.Fatal(err)
	}
	branch := &Qwen35SIMDBranch{packed: map[*tensor.Tensor][]float32{weight: packed}}
	if out%16 == 0 {
		// Fixed-topology constructor projections have no raw weight payload.
		// Tail-column cases below keep exercising the legacy raw fallback.
		branch.packedOnly = true
	}
	if withScratch {
		padded := qwen35ProjectionPaddedRows(rows)
		branch.scratch = map[string][]float32{
			"padIn":  make([]float32, padded*in),
			"padOut": make([]float32, padded*out),
		}
	}

	// Exercise the same reusable workers used by Forward, including tail work.
	if withScratch {
		stop := branch.startProjectionWorkers()
		defer stop()
	}

	inputStore := make([]float32, rows*in+2)
	inputStore[0], inputStore[len(inputStore)-1] = -12345.5, 98765.25
	x := inputStore[1 : 1+rows*in]
	dstStore := make([]float32, rows*out+2)
	dstStore[0], dstStore[len(dstStore)-1] = -2222.25, 3333.5
	dst := dstStore[1 : 1+rows*out]
	weightBefore := append([]float32(nil), weight.Data()...)
	for iter := 0; iter < 2; iter++ {
		for i := range x {
			x[i] = float32(((i+1)*(iter+2))%29-14) * 0.015625
		}
		wantInput := append([]float32(nil), x...)
		want := simdProjectionReference(x, weightBefore, rows, in, out)
		for i := range dst {
			dst[i] = float32(17-iter) * 1.5
		}
		var padInBefore, padOutBefore []float32
		if withScratch {
			padInBefore = fillSIMDBranchScratch(branch.scratch["padIn"], float32(iter)+11.5)
			padOutBefore = fillSIMDBranchScratch(branch.scratch["padOut"], float32(iter)+23.5)
		}
		if err := branch.project(dst, x, weight, rows, in, out); err != nil {
			t.Fatal(err)
		}
		requireProjectionClose(t, dst, want, rows, out)
		requireExactFloat32Slice(t, "input", x, wantInput)
		requireExactFloat32Slice(t, "weights", weight.Data(), weightBefore)
		if inputStore[0] != -12345.5 || inputStore[len(inputStore)-1] != 98765.25 {
			t.Fatal("input guard overwritten")
		}
		if dstStore[0] != -2222.25 || dstStore[len(dstStore)-1] != 3333.5 {
			t.Fatal("destination guard overwritten")
		}
		if withScratch {
			checkSIMDBranchScratch(t, branch, x, want, rows, in, out, padInBefore, padOutBefore)
		}
	}
	if withScratch {
		if allocs := testing.AllocsPerRun(10, func() {
			if err := branch.project(dst, x, weight, rows, in, out); err != nil {
				panic(err)
			}
		}); allocs != 0 {
			t.Fatalf("warm projection allocates: %g", allocs)
		}
	}
}

func TestSIMDBranchPackedOnlyProjection(t *testing.T) {
	oldProcs := runtime.GOMAXPROCS(6)
	defer runtime.GOMAXPROCS(oldProcs)
	oldAsm := simd.HasSgemmAsm
	defer func() { simd.HasSgemmAsm = oldAsm }()
	for _, enabled := range []bool{false, oldAsm} {
		simd.HasSgemmAsm = enabled
		for _, rows := range []int{1, 5, 6, 7, 12, 13} {
			for _, out := range []int{16, 512, 528} {
				for _, scratch := range []bool{false, true} {
					t.Run(fmt.Sprintf("asm%t/rows%d/out%d/scratch%t", enabled, rows, out, scratch), func(t *testing.T) {
						runSIMDBranchProjectionCase(t, rows, 9, out, scratch)
					})
				}
			}
		}
	}
}

func fillSIMDBranchScratch(buf []float32, base float32) []float32 {
	for i := range buf {
		buf[i] = base + float32((i%7)-3)
	}
	return append([]float32(nil), buf...)
}

func checkSIMDBranchScratch(t *testing.T, branch *Qwen35SIMDBranch, x, want []float32, rows, in, out int, padInBefore, padOutBefore []float32) {
	t.Helper()
	padIn, padOut := branch.scratch["padIn"], branch.scratch["padOut"]
	if rows%simd.SgemmNTRowBlock == 0 {
		requireExactFloat32Slice(t, "padIn", padIn, padInBefore)
		requireExactFloat32Slice(t, "padOut", padOut, padOutBefore)
		return
	}
	padded := qwen35ProjectionPaddedRows(rows)
	requireExactFloat32Slice(t, "padIn used rows", padIn[:rows*in], x)
	requireProjectionClose(t, padOut[:rows*out], want, rows, out)
	for i, v := range padIn[rows*in : padded*in] {
		if v != 0 {
			t.Fatalf("padIn tail[%d]=%g want 0", i, v)
		}
	}
	for i, v := range padOut[rows*out : padded*out] {
		if v != 0 {
			t.Fatalf("padOut tail[%d]=%g want 0", i, v)
		}
	}
}

func simdProjectionReference(x, weights []float32, rows, in, out int) []float32 {
	got := make([]float32, rows*out)
	for r := 0; r < rows; r++ {
		for c := 0; c < out; c++ {
			var sum float32
			for j := 0; j < in; j++ {
				sum += x[r*in+j] * weights[c*in+j]
			}
			got[r*out+c] = sum
		}
	}
	return got
}

func requireProjectionClose(t *testing.T, got, want []float32, rows, out int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > simdProjectionTol {
			t.Fatalf("projection mismatch row=%d col=%d got=%g want=%g", i/out, i%out, got[i], want[i])
		}
	}
}

func requireExactFloat32Slice(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len=%d want %d", name, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s[%d]=%g want %g", name, i, got[i], want[i])
		}
	}
}

func TestSIMDBranchConstructionRejects(t *testing.T) {
	for _, cap := range []int{-1, 0, 2, 513} {
		if s, e := NewQwen35SIMDBranch(nil, cfg.QwenNativeMTPMetadata{}, cap); e == nil || s != nil {
			t.Fatal("nil model")
		}
	}
	var s *Qwen35SIMDBranch
	if out, e := s.Forward(nil, 1, 1, nil, 1e-6); e == nil || out != nil {
		t.Fatal("nil executor")
	}
	// Validate before scratch access or matrix work.
	s = &Qwen35SIMDBranch{maxTokens: 8}
	for _, in := range [][][]float32{nil, {{0}}, {{0}, {0}, {0}}} {
		if out, e := s.Forward(in, 1, 1, nil, 1e-6); e == nil || out != nil {
			t.Fatal("invalid input")
		}
	}
}

func TestSIMDDeltaRowMatchesReference(t *testing.T) {
	a, b, q, k := make([]float32, 128), make([]float32, 128), make([]float32, 128), make([]float32, 128)
	for i := range q {
		q[i] = float32(i%11-5) * 0.02
		k[i] = float32(i%7-3) * 0.03
	}
	for i := 0; i < 64; i++ {
		v := float32(i%9-4) * 0.2
		want := qwen35LinearDeltaRowInPlace(a, q, k, v, 0.7, 0.95, 0.0883883476)
		simd.VecScale(b, b, 0.95)
		memory := simd.Sdot(b, k)
		simd.Saxpy((v-memory)*0.7, k, b)
		got := simd.Sdot(b, q) * 0.0883883476
		if math.Abs(float64(want-got)) > 1e-6 {
			t.Fatal("delta drift", i, want, got)
		}
	}
}
