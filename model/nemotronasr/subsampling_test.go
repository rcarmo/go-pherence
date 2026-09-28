package nemotronasr

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func releasedSubsampling(t testing.TB) *Subsampling {
	t.Helper()
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	model, loadErr := LoadSubsampling(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	return model
}

func compareASRStage(t *testing.T, name string, got []float32, expected int) {
	t.Helper()
	ref := readStemFixture(t, name, expected)
	if len(got) != len(ref) {
		t.Fatalf("%s len=%d want=%d", name, len(got), len(ref))
	}
	var maxAbs, sumAbs float64
	var outside int
	for i, v := range got {
		d := math.Abs(float64(v - ref[i]))
		if d > maxAbs {
			maxAbs = d
		}
		sumAbs += d
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || d > 3e-4+2e-5*math.Abs(float64(ref[i])) {
			outside++
		}
	}
	mean := sumAbs / float64(len(got))
	t.Logf("%s max_abs=%g mean_abs=%g outside=%d", name, maxAbs, mean, outside)
	meanLimit := 2e-5
	if len(got) == 5*1024 {
		// The 4352-term projection amplifies small differences in the stage
		// activations. This bound covers measured F32 accumulation error.
		meanLimit = 1e-4
	}
	if outside != 0 || mean > meanLimit {
		t.Fatalf("%s differs from independent reference distribution", name)
	}
}

func TestReleasedSubsamplingPyTorchParity(t *testing.T) {
	model := releasedSubsampling(t)
	features := readStemFixture(t, "features", 32*128)
	original := append([]float32(nil), features...)
	stem, err := model.Stem.ForwardOffline(features, 32)
	if err != nil {
		t.Fatal(err)
	}
	maskActivate(stem, 17, 65, 17)
	compareASRStage(t, "stem_activated", stem, 256*17*65)
	stage0, r0, w0, v0, err := model.layers[0].forward(stem, 17, 65, 17)
	if err != nil {
		t.Fatal(err)
	}
	if r0 != 9 || w0 != 33 || v0 != 9 {
		t.Fatalf("stage0 geometry %d %d %d", r0, w0, v0)
	}
	compareASRStage(t, "stage0_activated", stage0, 256*9*33)
	stage1, r1, w1, v1, err := model.layers[1].forward(stage0, r0, w0, v0)
	if err != nil {
		t.Fatal(err)
	}
	if r1 != 5 || w1 != 17 || v1 != 5 {
		t.Fatalf("stage1 geometry %d %d %d", r1, w1, v1)
	}
	compareASRStage(t, "stage1_activated", stage1, 256*5*17)
	out, err := model.ForwardOffline(features, 32, 32)
	if err != nil {
		t.Fatal(err)
	}
	compareASRStage(t, "projected", out, 5*1024)
	for _, valid := range []int{0, 1, 16, 31} {
		masked, err := model.ForwardOffline(features, 32, valid)
		if err != nil {
			t.Fatal(err)
		}
		compareASRStage(t, fmt.Sprintf("projected_valid%d", valid), masked, 5*1024)
	}
	for i, v := range features {
		if v != original[i] {
			t.Fatalf("caller feature mutated %d", i)
		}
	}
	stable := append([]float32(nil), out...)
	features[0]++
	if _, err := model.ForwardOffline(features, 32, 32); err != nil {
		t.Fatal(err)
	}
	for i, v := range out {
		if v != stable[i] {
			t.Fatalf("returned embedding changed %d", i)
		}
	}
}

func TestSubsamplingRejectsMalformed(t *testing.T) {
	model := &Subsampling{Stem: &StemConv2D{Weight: make([]float32, 256*9), Bias: make([]float32, 256)}, linearWeight: make([]float32, 1024*4352), linearBias: make([]float32, 1024)}
	for _, tc := range []struct {
		frames, valid int
		features      []float32
	}{{0, 0, nil}, {129, 128, make([]float32, 129*128)}, {1, -1, make([]float32, 128)}, {1, 2, make([]float32, 128)}, {2, 1, make([]float32, 127)}, {1, 1, []float32{float32(math.NaN())}}} {
		if _, err := model.ForwardOffline(tc.features, tc.frames, tc.valid); err == nil {
			t.Fatalf("accepted malformed input frames=%d valid=%d", tc.frames, tc.valid)
		}
	}
}

func BenchmarkReleasedSubsampling32(b *testing.B) {
	model := releasedSubsampling(b)
	features := readStemFixture(b, "features", 32*128)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := model.ForwardOffline(features, 32, 32); err != nil {
			b.Fatal(err)
		}
	}
}
