package main

import (
	"fmt"
	"time"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
)

func main() {
	fmt.Println("=== AI Worker Pool Benchmark ===")
	pool := ime2.NewAIWorkerPool(8)
	defer pool.Close()

	M, K := 2048, 2048
	wI8 := make([]int8, M*K)
	for i := range wI8 { wI8[i] = int8(i%15 + 1) }
	wPacked := ime2.PackTiles1024(wI8, M, K)
	actI8 := make([]int8, K)
	for i := range actI8 { actI8[i] = 10 }
	actPacked := ime2.BroadcastPack1024(actI8, K)
	out := make([]float32, M)

	// Correctness
	ime2.GemmAIPooled(M, K, wPacked, actPacked, 1.0, 1.0, out, pool)
	expected := float32(1 * 10 * K)
	fmt.Printf("Correctness: out[0]=%.0f expected=%.0f %s\n", out[0], expected,
		func() string { if out[0] == expected { return "✓" }; return "✗" }())

	// Benchmark: various sizes
	for _, size := range [][2]int{{2048, 2048}, {5632, 2048}, {2048, 5632}, {32000, 2048}} {
		m, k := size[0], size[1]
		w := ime2.PackTiles1024(make([]int8, m*k), m, k)
		a := ime2.BroadcastPack1024(make([]int8, k), k)
		for i := range w { w[i] = 5 }
		for i := range a { a[i] = 10 }
		o := make([]float32, m)

		const iters = 200
		t0 := time.Now()
		for i := 0; i < iters; i++ {
			ime2.GemmAIPooled(m, k, w, a, 1.0, 1.0, o, pool)
		}
		elapsed := time.Since(t0)
		usPerCall := float64(elapsed.Microseconds()) / float64(iters)
		gops := float64(m) * float64(k) * 4 * 2 / (usPerCall * 1000)
		fmt.Printf("  %5dx%4d (8 AI cores): %7.1f µs  %5.1f GOPS\n", m, k, usPerCall, gops)
	}
}
