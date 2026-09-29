package nemotrondiarization

import (
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func releasedLayer0Complete(t testing.TB) *Layer0Complete {
	t.Helper()
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m, loadErr := LoadLayer0Complete(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	return m
}

func TestReleasedLayer0CompletePyTorchParity(t *testing.T) {
	m := releasedLayer0Complete(t)
	full := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*512)
	for _, tc := range []struct {
		rows, count int
		name        string
	}{{16, 16 * 512, "testdata/jfk_layer0_complete.f32.gz"}, {138, 138 * 512, "testdata/jfk_full_layer0_complete.f32.gz"}} {
		input := full[:tc.count]
		original := append([]float32(nil), input...)
		got, err := m.ForwardOffline(input, tc.rows)
		if err != nil {
			t.Fatal(err)
		}
		ref := readStackingFixture(t, tc.name, tc.count)
		var maxAbs, sumAbs float64
		var outside int
		for i, value := range got {
			delta := math.Abs(float64(value - ref[i]))
			maxAbs = math.Max(maxAbs, delta)
			sumAbs += delta
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(ref[i])) {
				outside++
			}
		}
		mean := sumAbs / float64(len(got))
		t.Logf("layer0 complete rows=%d max_abs=%g mean_abs=%g outside=%d", tc.rows, maxAbs, mean, outside)
		if outside != 0 || mean > 2e-5 {
			t.Fatalf("rows=%d outside independent reference distribution", tc.rows)
		}
		for i := range input {
			if input[i] != original[i] {
				t.Fatalf("mutated caller input rows=%d i=%d", tc.rows, i)
			}
		}
		stable := got[0]
		input[0]++
		if got[0] != stable {
			t.Fatal("output aliases caller input")
		}
		input[0] = original[0]
	}
}

func TestLayer0CompleteRejectsMalformed(t *testing.T) {
	if _, err := LoadLayer0Complete(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
	m := &Layer0Complete{}
	if _, err := m.ForwardOffline(make([]float32, 512), 1); err == nil {
		t.Fatal("accepted missing weights")
	}
}

func BenchmarkReleasedLayer0Complete138(b *testing.B) {
	m := releasedLayer0Complete(b)
	input := readStackingFixture(b, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*512)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := m.ForwardOffline(input, 138); err != nil {
			b.Fatal(err)
		}
	}
}
