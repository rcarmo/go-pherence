package nemotronasr

import (
	"math"
	"os"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedEncoder0ConvolutionPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m, loadErr := LoadEncoder0Convolution(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	input := readStemFixture(t, "encoder0_attn_residual", 5*encoderWidth)
	original := append([]float32(nil), input...)
	normal := make([]float32, len(input))
	if !simd.LayerNormLastAxisTo(normal, input, 5, encoderWidth, m.preGamma, m.preBeta, 1e-5) {
		t.Fatal("pre-convolution normalisation rejected shape")
	}
	compareASRStage(t, "encoder0_conv_normal", normal, len(input))
	point := make([]float32, 5*2*encoderWidth)
	if !simd.DenseNTTo(point, normal, m.point1, 5, 2*encoderWidth, encoderWidth, 1, encoderWidth, encoderWidth, 2*encoderWidth) {
		t.Fatal("pointwise1 rejected shape")
	}
	compareConvChannelMajor(t, "encoder0_conv_point1", point, 5, 2*encoderWidth)
	glu := make([]float32, len(input))
	for row := 0; row < 5; row++ {
		for ch := 0; ch < encoderWidth; ch++ {
			gate := point[row*2*encoderWidth+encoderWidth+ch]
			glu[row*encoderWidth+ch] = point[row*2*encoderWidth+ch] / (1 + float32(math.Exp(float64(-gate))))
		}
	}
	compareConvChannelMajor(t, "encoder0_conv_glu", glu, 5, encoderWidth)
	depth := make([]float32, len(input))
	for row := 0; row < 5; row++ {
		for ch := 0; ch < encoderWidth; ch++ {
			for tap := 0; tap < encoderConvKernel; tap++ {
				source := row + tap - (encoderConvKernel - 1)
				if source >= 0 {
					depth[row*encoderWidth+ch] += glu[source*encoderWidth+ch] * m.depth[ch*encoderConvKernel+tap]
				}
			}
		}
	}
	compareConvChannelMajor(t, "encoder0_conv_depth", depth, 5, encoderWidth)
	depthNormal := make([]float32, len(input))
	if !simd.LayerNormLastAxisTo(depthNormal, depth, 5, encoderWidth, m.depthGamma, m.depthBeta, 1e-5) {
		t.Fatal("depth normalisation rejected shape")
	}
	compareASRStage(t, "encoder0_conv_depth_normal", depthNormal, len(input))
	if !simd.SiLUTo(depthNormal, depthNormal) {
		t.Fatal("depth activation rejected shape")
	}
	compareConvChannelMajor(t, "encoder0_conv_activated", depthNormal, 5, encoderWidth)
	output, residual, err := m.ForwardOffline(input, 5)
	if err != nil {
		t.Fatal(err)
	}
	compareASRStage(t, "encoder0_conv_output", output, 5*encoderWidth)
	compareASRStage(t, "encoder0_conv_residual", residual, 5*encoderWidth)
	for i, value := range input {
		if value != original[i] {
			t.Fatalf("mutated caller input %d", i)
		}
	}
	if &output[0] == &residual[0] || &output[0] == &input[0] || &residual[0] == &input[0] {
		t.Fatal("convolution outputs alias caller or one another")
	}
}

func compareConvChannelMajor(t *testing.T, name string, rowMajor []float32, rows, channels int) {
	t.Helper()
	ref := readStemFixture(t, name, rows*channels)
	var maxAbs, sumAbs float64
	var outside int
	for row := 0; row < rows; row++ {
		for ch := 0; ch < channels; ch++ {
			actual := rowMajor[row*channels+ch]
			want := ref[ch*rows+row]
			delta := math.Abs(float64(actual - want))
			maxAbs = math.Max(maxAbs, delta)
			sumAbs += delta
			if math.IsNaN(float64(actual)) || math.IsInf(float64(actual), 0) || delta > 3e-4+2e-5*math.Abs(float64(want)) {
				outside++
			}
		}
	}
	mean := sumAbs / float64(len(rowMajor))
	t.Logf("%s max_abs=%g mean_abs=%g outside=%d", name, maxAbs, mean, outside)
	if outside != 0 || mean > 2e-5 {
		t.Fatalf("%s differs from independent PyTorch fixture", name)
	}
}

func TestEncoder0ConvolutionRejectsMalformed(t *testing.T) {
	if _, err := LoadEncoder0Convolution(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
	m := &Encoder0Convolution{}
	if _, _, err := m.ForwardOffline(make([]float32, encoderWidth), 1); err == nil {
		t.Fatal("accepted missing weights")
	}
	m = &Encoder0Convolution{preGamma: make([]float32, encoderWidth), preBeta: make([]float32, encoderWidth), point1: make([]float32, 2*encoderWidth*encoderWidth), depth: make([]float32, encoderWidth*encoderConvKernel), depthGamma: make([]float32, encoderWidth), depthBeta: make([]float32, encoderWidth), point2: make([]float32, encoderWidth*encoderWidth)}
	for _, rows := range []int{0, 6} {
		if _, _, err := m.ForwardOffline(make([]float32, rows*encoderWidth), rows); err == nil {
			t.Fatalf("accepted rows=%d", rows)
		}
	}
	if _, _, err := m.ForwardOffline(make([]float32, encoderWidth-1), 1); err == nil {
		t.Fatal("accepted short input")
	}
	invalid := make([]float32, encoderWidth)
	invalid[0] = float32(math.NaN())
	if _, _, err := m.ForwardOffline(invalid, 1); err == nil {
		t.Fatal("accepted non-finite input")
	}
}

func BenchmarkReleasedEncoder0Convolution5(b *testing.B) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		b.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	m, loadErr := LoadEncoder0Convolution(file)
	if err := file.Close(); err != nil {
		b.Fatal(err)
	}
	if loadErr != nil {
		b.Fatal(loadErr)
	}
	input := readStemFixture(b, "encoder0_attn_residual", 5*encoderWidth)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, _, err := m.ForwardOffline(input, 5); err != nil {
			b.Fatal(err)
		}
	}
}
