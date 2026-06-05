//go:build riscv64

package simd

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

func TestSdotM4Correct(t *testing.T) {
	if !HasDotAsm {
		t.Skip("no RVV dot")
	}
	rng := rand.New(rand.NewSource(3))
	for _, k := range []int{1, 2, 7, 8, 9, 31, 32, 33, 80, 127, 256, 1280, 1281} {
		x := make([]float32, k)
		y := make([]float32, k)
		var truth float64
		for i := 0; i < k; i++ {
			x[i] = rng.Float32()*2 - 1
			y[i] = rng.Float32()*2 - 1
			truth += float64(x[i]) * float64(y[i])
		}
		got := sdotM4Asm(x, y)
		tol := 1e-3 + 1e-6*float64(k)
		if math.Abs(float64(got)-truth) > tol {
			t.Fatalf("k=%d sdotM4=%v truth=%v", k, got, truth)
		}
		got8 := sdotM8Asm(x, y)
		if math.Abs(float64(got8)-truth) > tol {
			t.Fatalf("k=%d sdotM8=%v truth=%v", k, got8, truth)
		}
		gotx2 := sdotM4x2Asm(x, y)
		if math.Abs(float64(gotx2)-truth) > tol {
			t.Fatalf("k=%d sdotM4x2=%v truth=%v", k, gotx2, truth)
		}
	}
}

func TestSdotM4Speed(t *testing.T) {
	if !HasDotAsm {
		t.Skip("no RVV dot")
	}
	const k = 1280
	const iters = 200000
	x := make([]float32, k)
	y := make([]float32, k)
	for i := 0; i < k; i++ {
		x[i] = float32(i%17) * 0.1
		y[i] = float32(i%13) * 0.1
	}

	// warm
	_ = sdotAsm(x, y)
	_ = sdotM4Asm(x, y)
	_ = sdotM8Asm(x, y)

	var s float32
	t0 := time.Now()
	for i := 0; i < iters; i++ {
		s += sdotAsm(x, y)
	}
	m1 := time.Since(t0)

	t0 = time.Now()
	for i := 0; i < iters; i++ {
		s += sdotM4Asm(x, y)
	}
	m4 := time.Since(t0)

	t0 = time.Now()
	for i := 0; i < iters; i++ {
		s += sdotM8Asm(x, y)
	}
	m8 := time.Since(t0)

	t0 = time.Now()
	for i := 0; i < iters; i++ {
		s += sdotM4x2Asm(x, y)
	}
	m4x2 := time.Since(t0)

	flops := 2.0 * float64(k) * float64(iters)
	t.Logf("m1   sdot: %v  (%.2f GFLOP/s)", m1, flops/m1.Seconds()/1e9)
	t.Logf("m4   sdot: %v  (%.2f GFLOP/s)", m4, flops/m4.Seconds()/1e9)
	t.Logf("m8   sdot: %v  (%.2f GFLOP/s)", m8, flops/m8.Seconds()/1e9)
	t.Logf("m4x2 sdot: %v  (%.2f GFLOP/s)", m4x2, flops/m4x2.Seconds()/1e9)
	t.Logf("m4/m1: %.2fx  m4x2/m4: %.2fx  (sink=%v)", float64(m1)/float64(m4), float64(m4)/float64(m4x2), s)
}
