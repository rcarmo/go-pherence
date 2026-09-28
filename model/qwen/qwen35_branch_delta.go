package qwen

import (
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// qwen35BranchDeltaHead updates one fixed 128x128 recurrent head. Inputs are
// internal validated scratch: state/output are disjoint from q/k/v. Four-row
// dot products keep each row's two-accumulator reduction on amd64; other hosts
// and assembly-disabled callers retain the row-by-row implementation.
func qwen35BranchDeltaHead(state, out, q, k, v []float32, decay, beta float32) {
	const scale = float32(1 / math.Sqrt2 / 8) // 1/sqrt(128)
	if simd.HasSdotx4SIMD && simd.HasDotAsm {
		for j := 0; j < 128; j += 4 {
			rows := state[j*128 : (j+4)*128]
			simd.VecScale(rows, rows, decay)
			m0, m1, m2, m3, _ := simd.Sdotx4(k, rows, 128)
			memory := [4]float32{m0, m1, m2, m3}
			for n := 0; n < 4; n++ {
				simd.Saxpy((v[j+n]-memory[n])*beta, k, rows[n*128:(n+1)*128])
			}
			v0, v1, v2, v3, _ := simd.Sdotx4(q, rows, 128)
			values := [4]float32{v0, v1, v2, v3}
			for n := 0; n < 4; n++ {
				out[j+n] = values[n] * scale
			}
		}
		return
	}
	for j := 0; j < 128; j++ {
		row := state[j*128 : (j+1)*128]
		simd.VecScale(row, row, decay)
		memory := simd.Sdot(row, k)
		simd.Saxpy((v[j]-memory)*beta, k, row)
		out[j] = simd.Sdot(row, q) * scale
	}
}
