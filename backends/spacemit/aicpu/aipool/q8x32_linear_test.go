package aipool

import (
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
)

func TestGemmQ80x32AIPooledSmoke(t *testing.T) {
	if _, err := os.Stat("/proc/set_ai_thread"); err != nil {
		t.Skip("no A100 thread registration")
	}
	const M, N, K = 8, 32, 64
	x := make([]float32, M*K)
	w := make([]float32, N*K)
	for i := range x {
		x[i] = float32((i%19)-9) / 9
	}
	for i := range w {
		w[i] = float32((i%17)-8) / 8
	}
	out := make([]float32, M*N)
	pool := NewAIWorkerPool(2)
	defer pool.Close()
	wq := ime2.PackF32ToQ80x32(N, K, w)
	if !GemmQ80x32AIPooled(x, M, K, wq, out, pool) {
		t.Fatal("GemmQ80x32AIPooled returned false")
	}
	// Check against f32 matmul with a loose tolerance: this path quantizes both
	// activations and weights to Q8_0, so exact f32 equality is not expected.
	maxDiff := float32(0)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var ref float32
			for k := 0; k < K; k++ {
				ref += x[m*K+k] * w[n*K+k]
			}
			d := float32(math.Abs(float64(out[m*N+n] - ref)))
			if d > maxDiff {
				maxDiff = d
			}
		}
	}
	if maxDiff > 1.0 {
		t.Fatalf("max diff too large: %.4f", maxDiff)
	}
}

func TestGemmQ80x32AIPooledX100PackMatchesWorkerPack(t *testing.T) {
	if _, err := os.Stat("/proc/set_ai_thread"); err != nil {
		t.Skip("no A100 thread registration")
	}
	const M, N, K = 7, 32, 64
	x := make([]float32, M*K)
	w := make([]float32, N*K)
	for i := range x {
		x[i] = float32((i%17)-8) / 9
	}
	for i := range w {
		w[i] = float32((i%19)-9) / 11
	}
	wq := ime2.PackF32ToQ80x32(N, K, w)
	pool := NewAIWorkerPool(2)
	defer pool.Close()
	for _, tc := range []struct {
		name string
		base func([]float32, int, int, ime2.Q80x32, []float32, *AIWorkerPool) bool
		x100 func([]float32, int, int, ime2.Q80x32, []float32, *AIWorkerPool) bool
	}{
		{"plain", GemmQ80x32AIPooled, GemmQ80x32AIPooledX100Pack},
		{"gelu", GemmQ80x32AIPooledGELU, GemmQ80x32AIPooledGELUX100Pack},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want := make([]float32, M*N)
			got := make([]float32, M*N)
			if !tc.base(x, M, K, wq, want, pool) {
				t.Fatal("base pack failed")
			}
			if !tc.x100(x, M, K, wq, got, pool) {
				t.Fatal("x100 pack failed")
			}
			maxDiff := float32(0)
			for i := range want {
				d := want[i] - got[i]
				if d < 0 {
					d = -d
				}
				if d > maxDiff {
					maxDiff = d
				}
			}
			if maxDiff != 0 {
				t.Fatalf("maxDiff=%g", maxDiff)
			}
		})
	}
}
