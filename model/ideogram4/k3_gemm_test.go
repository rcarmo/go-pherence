package ideogram4

import "testing"

func TestK3GemmRowsF32FallbackOffHost(t *testing.T) {
	out := make([]float32, 4)
	a := []float32{1, 2, 3, 4}
	b := []float32{1, 0, 0, 1}
	if ok := k3GemmRowsF32(out, a, b, 2, 2, 2); ok {
		// A riscv64 test environment may opt into GO_PHERENCE_IDEOGRAM4_K3 globally.
		// In that case just ensure the bridge produced the expected finite matrix.
		want := []float32{1, 2, 3, 4}
		for i := range want {
			if out[i] != want[i] {
				t.Fatalf("out[%d]=%v want %v", i, out[i], want[i])
			}
		}
	}
}
