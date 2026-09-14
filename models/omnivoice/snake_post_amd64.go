//go:build amd64

package omnivoice

import simd "github.com/rcarmo/go-pherence/backends/simd/runtime"

//go:noescape
func snakePostAsm(row, sine []float32, scale float32)

func snakePost(row, sine []float32, scale float32) {
	if simd.HasVecAsm {
		snakePostAsm(row, sine, scale)
		return
	}
	snakePostVectors(row, sine, scale)
}
