package qwen

import (
	"math"
	"runtime"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// Reference is the row-at-a-time loop used before four-row grouping. Dispatch
// flags stay the same for both sides so this checks ordering, not scalar/FMA
// approximation. No goroutine changes package dispatch flags concurrently.
func deltaHeadRowReference(state, out, q, k, v []float32, decay, beta float32) {
	for j := 0; j < 128; j++ {
		row := state[j*128 : (j+1)*128]
		simd.VecScale(row, row, decay)
		memory := simd.Sdot(row, k)
		simd.Saxpy((v[j]-memory)*beta, k, row)
		out[j] = simd.Sdot(row, q) * float32(1/math.Sqrt(128))
	}
}

func TestBranchDeltaHeadExact(t *testing.T) {
	oldDot, oldVec := simd.HasDotAsm, simd.HasVecAsm
	defer func() { simd.HasDotAsm, simd.HasVecAsm = oldDot, oldVec }()
	for _, mode := range []string{"native", "no-dot", "scalar"} {
		t.Run(mode, func(t *testing.T) {
			simd.HasDotAsm, simd.HasVecAsm = oldDot, oldVec
			if mode == "no-dot" {
				simd.HasDotAsm = false
			}
			if mode == "scalar" {
				simd.HasDotAsm, simd.HasVecAsm = false, false
			}
			stateStore, outStore := make([]float32, 128*128+2), make([]float32, 130)
			stateStore[0], stateStore[len(stateStore)-1] = 77, -99
			outStore[0], outStore[129] = 33, -44
			state, out := stateStore[1:len(stateStore)-1], outStore[1:129]
			q, k, v := make([]float32, 128), make([]float32, 128), make([]float32, 128)
			seed := uint32(17)
			fill := func(x []float32) {
				for i := range x {
					seed = seed*1664525 + 1013904223
					x[i] = float32(int32(seed>>12)-524288) / 16777216
				}
			}
			fill(state)
			fill(q)
			fill(k)
			fill(v)
			wantState, want := append([]float32(nil), state...), make([]float32, 128)
			qCopy, kCopy, vCopy := append([]float32(nil), q...), append([]float32(nil), k...), append([]float32(nil), v...)
			for step := 0; step < 12; step++ {
				decay, beta := float32(step%3)*0.375, float32(step%4)*0.25
				deltaHeadRowReference(wantState, want, q, k, v, decay, beta)
				qwen35BranchDeltaHead(state, out, q, k, v, decay, beta)
				for i, x := range state {
					if math.Float32bits(x) != math.Float32bits(wantState[i]) {
						t.Fatalf("state step%d index%d got%x want%x", step, i, math.Float32bits(x), math.Float32bits(wantState[i]))
					}
				}
				for i, x := range out {
					if math.Float32bits(x) != math.Float32bits(want[i]) {
						t.Fatalf("out step%d index%d got%x want%x", step, i, math.Float32bits(x), math.Float32bits(want[i]))
					}
				}
			}
			requireExactFloat32Slice(t, "q", q, qCopy)
			requireExactFloat32Slice(t, "k", k, kCopy)
			requireExactFloat32Slice(t, "v", v, vCopy)
			if stateStore[0] != 77 || stateStore[len(stateStore)-1] != -99 || outStore[0] != 33 || outStore[129] != -44 {
				t.Fatal("guard overwritten")
			}
			if runtime.GOARCH != "riscv64" || mode == "scalar" {
				if n := testing.AllocsPerRun(10, func() { qwen35BranchDeltaHead(state, out, q, k, v, .5, .5) }); n != 0 {
					t.Fatalf("allocations %g", n)
				}
			}
		})
	}
}

func BenchmarkBranchDeltaHead(b *testing.B) {
	for _, name := range []string{"rows", "four"} {
		b.Run(name, func(b *testing.B) {
			state, out := make([]float32, 128*128), make([]float32, 128)
			q, k, v := make([]float32, 128), make([]float32, 128), make([]float32, 128)
			for i := range q {
				q[i], k[i], v[i] = .03125, .03125, .125
			}
			fn := deltaHeadRowReference
			if name == "four" {
				fn = qwen35BranchDeltaHead
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				fn(state, out, q, k, v, .75, .5)
			}
		})
	}
}
