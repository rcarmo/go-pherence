package ime2

import (
	"fmt"
	"testing"
	"time"
)

func TestGemmSpeedupSizes(t *testing.T) {
	for _, size := range [][3]int{{2048,4,2048}, {2048,4,1024}, {4096,4,1024}, {1024,4,4096}} {
		M, N, K := size[0], size[1], size[2]
		Ap := PackTiles(make([]int8, M*K), M, K)
		Bp := PackTiles(make([]int8, N*K), N, K)
		C := make([]int32, M*N)
		const iters = 30
		t0 := time.Now()
		for i := 0; i < iters; i++ { GemmINT8Packed(M, N, K, Ap, Bp, C) }
		single := time.Since(t0) / iters
		t1 := time.Now()
		for i := 0; i < iters; i++ { GemmINT8PackedParallel(M, N, K, Ap, Bp, C, 8) }
		parallel := time.Since(t1) / iters
		fmt.Printf("%dx%dx%d: single=%v par=%v speedup=%.2fx\n", M, N, K, single, parallel, float64(single)/float64(parallel))
	}
}

func TestGemmFFNSizes(t *testing.T) {
	// TinyLlama FFN dimensions
	for _, size := range [][3]int{{5632,4,2048}, {2048,4,5632}} {
		M, N, K := size[0], size[1], size[2]
		Ap := PackTiles(make([]int8, M*K), M, K)
		Bp := PackTiles(make([]int8, N*K), N, K)
		C := make([]int32, M*N)
		const iters = 20
		t0 := time.Now()
		for i := 0; i < iters; i++ { GemmINT8Packed(M, N, K, Ap, Bp, C) }
		single := time.Since(t0) / iters
		t1 := time.Now()
		for i := 0; i < iters; i++ { GemmINT8PackedParallel(M, N, K, Ap, Bp, C, 8) }
		parallel := time.Since(t1) / iters
		fmt.Printf("FFN %dx%dx%d: single=%v par=%v speedup=%.2fx\n", M, N, K, single, parallel, float64(single)/float64(parallel))
	}
}
