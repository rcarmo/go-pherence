package main

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"syscall"
	"time"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
)

func main() {
	runtime.LockOSThread()
	f, _ := os.OpenFile("/proc/set_ai_thread", os.O_WRONLY, 0)
	f.Write([]byte(strconv.Itoa(syscall.Gettid())))
	f.Close()
	fmt.Println("=== AI Core GEMM Benchmark (VLEN=1024) ===")

	M, K := 2048, 2048

	// Create weights (each row r has value (r%15)+1)
	wI8 := make([]int8, M*K)
	for r := 0; r < M; r++ {
		v := int8((r%15) + 1)
		for k := 0; k < K; k++ { wI8[r*K+k] = v }
	}
	wPacked := ime2.PackTiles1024(wI8, M, K)

	// Activation (all 10)
	actI8 := make([]int8, K)
	for i := range actI8 { actI8[i] = 10 }
	actPacked := ime2.BroadcastPack1024(actI8, K)

	// Run GEMM
	out := make([]float32, M)
	ime2.GemmINT8_AI(M, K, wPacked, actPacked, 1.0, 1.0, out)

	// Verify first few rows
	fmt.Println("Correctness check:")
	for r := 0; r < 8; r++ {
		expected := float32((r%15 + 1) * 10 * K)
		match := "✓"
		if out[r] != expected { match = fmt.Sprintf("✗ (got %.0f)", out[r]) }
		fmt.Printf("  row %d: expected=%.0f %s\n", r, expected, match)
	}

	// Benchmark
	fmt.Println("\nBenchmark (single AI core):")
	for _, size := range [][2]int{{2048, 2048}, {5632, 2048}, {2048, 5632}} {
		m, k := size[0], size[1]
		w := ime2.PackTiles1024(make([]int8, m*k), m, k)
		a := ime2.BroadcastPack1024(make([]int8, k), k)
		o := make([]float32, m)
		for i := range w { w[i] = 5 }
		for i := range a { a[i] = 10 }

		const iters = 500
		t0 := time.Now()
		for i := 0; i < iters; i++ {
			ime2.GemmINT8_AI(m, k, w, a, 1.0, 1.0, o)
		}
		elapsed := time.Since(t0)
		usPerCall := float64(elapsed.Microseconds()) / float64(iters)
		gops := float64(m) * float64(k) * 4 * 2 / (usPerCall * 1000)
		fmt.Printf("  %dx%d: %.1f µs/call  %.1f GOPS\n", m, k, usPerCall, gops)
	}
}
