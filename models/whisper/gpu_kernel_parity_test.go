package whisper

import (
	"math"
	"os"
	"testing"

	nv "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/audio"
)

func TestWhisperCUDAMelParity(t *testing.T) {
	if !nv.SgemmReady() {
		t.Skip("CUDA SGEMM not available")
	}
	cfg := LargeV3Turbo()
	samples := make([]float32, 16000/5)
	for i := range samples {
		// Deterministic, low-amplitude mixture that exercises multiple FFT bins
		// without clipping or denorm-heavy silence.
		samples[i] = 0.08*float32(math.Sin(2*math.Pi*440*float64(i)/16000)) + 0.03*float32(math.Sin(2*math.Pi*880*float64(i)/16000))
	}
	t.Setenv("GO_PHERENCE_WHISPER_GPU_MEL", "0")
	want, wantFrames := MelFlatFromSamples(samples, cfg)
	t.Setenv("GO_PHERENCE_WHISPER_GPU_MEL", "1")
	melCfg := audio.MelConfig{SampleRate: 16000, FFTSize: 400, HopLength: 160, NumMels: cfg.NumMelBins, NFFTPadded: 512}
	got, gotFrames, ok := melFlatGPU(samples, melCfg)
	if !ok {
		t.Fatalf("melFlatGPU returned fallback")
	}
	if gotFrames != wantFrames {
		t.Fatalf("frame mismatch: got=%d want=%d", gotFrames, wantFrames)
	}
	assertClose(t, got, want, 8e-2)
}

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

func TestWhisperCUDAEncoderGEMVParity(t *testing.T) {
	if !nv.SgemmReady() {
		t.Skip("CUDA SGEMM not available")
	}
	seqLen, inDim, outDim := 9, 7, 11
	x := make([]float32, seqLen*inDim)
	for i := range x {
		x[i] = float32((i%13)-6) * 0.037
	}
	_, weight, bias := deterministicLinearInputs(inDim, outDim)
	want := linearForwardOpt(x, weight, bias, seqLen, inDim, outDim)
	wDev := nv.NewDevBufFrom(weight)
	if err := wDev.ToGPU(); err != nil {
		t.Fatalf("weight ToGPU: %v", err)
	}
	got := gemvGPU(x, wDev, bias, seqLen, inDim, outDim)
	assertClose(t, got, want, 2e-4)
}

func TestWhisperCUDAEncoderAttentionParity(t *testing.T) {
	if !nv.SgemmReady() {
		t.Skip("CUDA SGEMM not available")
	}
	t.Setenv("GO_PHERENCE_WHISPER_GPU_ATTENTION", "1")
	seqLen, numHeads, headDim := 5, 2, 4
	q, k, v := deterministicQKV(seqLen, seqLen, numHeads, headDim)
	want := fullAttention(q, k, v, seqLen, seqLen, numHeads, headDim)
	got, ok := fullAttentionGPU(q, k, v, seqLen, numHeads, headDim)
	if !ok {
		t.Fatalf("fullAttentionGPU returned fallback")
	}
	assertClose(t, got, want, 3e-4)
}

func TestWhisperCUDADecoderSelfAttentionParity(t *testing.T) {
	if !nv.SgemmReady() {
		t.Skip("CUDA SGEMM not available")
	}
	t.Setenv("GO_PHERENCE_WHISPER_GPU_SELF_ATTN", "1")
	seqKV, numHeads, headDim := 6, 2, 4
	q, k, v := deterministicQKV(1, seqKV, numHeads, headDim)
	want := fullAttention(q, k, v, 1, seqKV, numHeads, headDim)
	got := make([]float32, numHeads*headDim)
	bufs := newDecoderBufs(Config{DecoderDModel: numHeads * headDim, EncoderFFNDim: 16})
	if !bufs.selfAttentionGPU(got, q, k, v, seqKV, numHeads, headDim) {
		t.Fatalf("selfAttentionGPU returned fallback")
	}
	assertClose(t, got, want, 3e-4)
}

func TestWhisperCUDADecoderCrossAttentionParity(t *testing.T) {
	if !nv.SgemmReady() {
		t.Skip("CUDA SGEMM not available")
	}
	seqKV, numHeads, headDim := 6, 2, 4
	q, k, v := deterministicQKV(1, seqKV, numHeads, headDim)
	want := fullAttention(q, k, v, 1, seqKV, numHeads, headDim)
	got := make([]float32, numHeads*headDim)
	kDev := nv.NewDevBufFrom(k)
	if err := kDev.ToGPU(); err != nil {
		t.Fatalf("k ToGPU: %v", err)
	}
	vDev := nv.NewDevBufFrom(v)
	if err := vDev.ToGPU(); err != nil {
		t.Fatalf("v ToGPU: %v", err)
	}
	bufs := newDecoderBufs(Config{DecoderDModel: numHeads * headDim, EncoderFFNDim: 16})
	if !bufs.attentionGPU(got, q, []*nv.DevBuf{kDev}, []*nv.DevBuf{vDev}, 0, seqKV, numHeads, headDim) {
		t.Fatalf("attentionGPU returned fallback")
	}
	assertClose(t, got, want, 3e-4)
}

func TestWhisperCUDADecoderLinearParity(t *testing.T) {
	if !nv.SgemmReady() {
		t.Skip("CUDA SGEMM not available")
	}
	inDim, outDim := 7, 11
	x, weight, bias := deterministicLinearInputs(inDim, outDim)
	want := make([]float32, outDim)
	linearInto(want, x, weight, bias, inDim, outDim)
	got := make([]float32, outDim)
	wDev := nv.NewDevBufFrom(weight)
	if err := wDev.ToGPU(); err != nil {
		t.Fatalf("weight ToGPU: %v", err)
	}
	bufs := newDecoderBufs(Config{DecoderDModel: inDim, EncoderFFNDim: outDim})
	if !bufs.linearGPU(got, x, wDev, bias, inDim, outDim) {
		t.Fatalf("linearGPU returned fallback")
	}
	assertClose(t, got, want, 2e-4)
}

func TestWhisperCUDALMHeadParity(t *testing.T) {
	if !nv.SgemmReady() {
		t.Skip("CUDA SGEMM not available")
	}
	h, vocab := 9, 13
	x, weight, _ := deterministicLinearInputs(h, vocab)
	want := make([]float32, vocab)
	linearInto(want, x, weight, nil, h, vocab)
	got := make([]float32, vocab)
	wDev := nv.NewDevBufFrom(weight)
	if err := wDev.ToGPU(); err != nil {
		t.Fatalf("weight ToGPU: %v", err)
	}
	bufs := newDecoderBufs(Config{DecoderDModel: h, EncoderFFNDim: vocab})
	if !bufs.lmHeadGPU(got, x, wDev, vocab, h) {
		t.Fatalf("lmHeadGPU returned fallback")
	}
	assertClose(t, got, want, 2e-4)
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
	if !UseGPULMHead() {
		t.Fatalf("GPU graph umbrella did not enable LM-head dispatch")
	}
	if !UseGPUCrossKV() {
		t.Fatalf("GPU graph umbrella did not enable cross-K/V dispatch")
	}
	if !UseGPUCrossAttention() {
		t.Fatalf("GPU graph umbrella did not enable cross-attention dispatch")
	}
}

func deterministicLinearInputs(inDim, outDim int) ([]float32, []float32, []float32) {
	x := make([]float32, inDim)
	weight := make([]float32, outDim*inDim)
	bias := make([]float32, outDim)
	for i := range x {
		x[i] = float32((i%11)-5) * 0.041
	}
	for i := range weight {
		weight[i] = float32((i%17)-8) * 0.029
	}
	for i := range bias {
		bias[i] = float32((i%7)-3) * 0.037
	}
	return x, weight, bias
}

func deterministicQKV(seqQ, seqKV, numHeads, headDim int) ([]float32, []float32, []float32) {
	q := make([]float32, seqQ*numHeads*headDim)
	k := make([]float32, seqKV*numHeads*headDim)
	v := make([]float32, seqKV*numHeads*headDim)
	for i := range q {
		q[i] = float32((i%17)-8) * 0.031
	}
	for i := range k {
		k[i] = float32((i%19)-9) * 0.027
	}
	for i := range v {
		v[i] = float32((i%23)-11) * 0.023
	}
	return q, k, v
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
