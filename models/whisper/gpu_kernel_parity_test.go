package whisper

import (
	"math"
	"os"
	"testing"

	nv "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func TestWhisperCUDAConv1DParity(t *testing.T) {
	if !nv.SgemmReady() {
		t.Skip("CUDA SGEMM not available")
	}
	t.Setenv("GO_PHERENCE_WHISPER_GPU_CONV1D", "1")
	for _, stride := range []int{1, 2} {
		stride := stride
		t.Run("stride", func(t *testing.T) {
			inCh, inLen, outCh := 3, 9, 4
			input := make([]float32, inCh*inLen)
			for i := range input {
				input[i] = float32((i%11)-5) * 0.07
			}
			weight := make([]float32, outCh*inCh*3)
			for i := range weight {
				weight[i] = float32((i%13)-6) * 0.03
			}
			bias := make([]float32, outCh)
			for i := range bias {
				bias[i] = float32(i-2) * 0.05
			}
			want := conv1dForward(input, weight, bias, inCh, inLen, outCh, 3, stride, 1)
			got, ok := conv1dForwardGPU(input, weight, bias, inCh, inLen, outCh, stride)
			if !ok {
				t.Fatalf("conv1dForwardGPU returned fallback")
			}
			assertClose(t, got, want, 2e-4)
		})
	}
}

func TestWhisperCUDAEncoderAttentionParity(t *testing.T) {
	if !nv.SgemmReady() {
		t.Skip("CUDA SGEMM not available")
	}
	t.Setenv("GO_PHERENCE_WHISPER_GPU_ATTENTION", "1")
	seqLen, numHeads, headDim := 5, 2, 4
	n := seqLen * numHeads * headDim
	q := make([]float32, n)
	k := make([]float32, n)
	v := make([]float32, n)
	for i := 0; i < n; i++ {
		q[i] = float32((i%17)-8) * 0.031
		k[i] = float32((i%19)-9) * 0.027
		v[i] = float32((i%23)-11) * 0.023
	}
	want := fullAttention(q, k, v, seqLen, seqLen, numHeads, headDim)
	got, ok := fullAttentionGPU(q, k, v, seqLen, numHeads, headDim)
	if !ok {
		t.Fatalf("fullAttentionGPU returned fallback")
	}
	assertClose(t, got, want, 3e-4)
}

func TestWhisperGPUGraphUmbrellaEnablesKernelDispatch(t *testing.T) {
	oldConv := os.Getenv("GO_PHERENCE_WHISPER_GPU_CONV1D")
	oldAttn := os.Getenv("GO_PHERENCE_WHISPER_GPU_ATTENTION")
	defer os.Setenv("GO_PHERENCE_WHISPER_GPU_CONV1D", oldConv)
	defer os.Setenv("GO_PHERENCE_WHISPER_GPU_ATTENTION", oldAttn)
	os.Unsetenv("GO_PHERENCE_WHISPER_GPU_CONV1D")
	os.Unsetenv("GO_PHERENCE_WHISPER_GPU_ATTENTION")
	t.Setenv("GO_PHERENCE_WHISPER_GPU_GRAPH", "1")
	if !whisperGPUFeatureEnabled("GO_PHERENCE_WHISPER_GPU_CONV1D") {
		t.Fatalf("GPU graph umbrella did not enable conv1d dispatch")
	}
	if !whisperGPUFeatureEnabled("GO_PHERENCE_WHISPER_GPU_ATTENTION") {
		t.Fatalf("GPU graph umbrella did not enable attention dispatch")
	}
}

func assertClose(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got=%d want=%d", len(got), len(want))
	}
	var maxDiff float64
	maxIdx := -1
	for i := range got {
		d := math.Abs(float64(got[i] - want[i]))
		if d > maxDiff {
			maxDiff = d
			maxIdx = i
		}
	}
	if maxDiff > tol {
		t.Fatalf("max diff %.6g at %d exceeds %.6g: got=%g want=%g", maxDiff, maxIdx, tol, got[maxIdx], want[maxIdx])
	}
}
