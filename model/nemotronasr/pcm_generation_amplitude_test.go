package nemotronasr

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/audio"
)

// Amplified recorded PCM is a bounded cross-backend numerical stress case.
// SIMD is an implementation comparison, not an independent labelled WER
// reference. We keep the existing subsampling and tower error gates.
func TestReleasedPCMGenerationAmplifiedBackendParity(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_ASR_AMPLITUDE") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_ASR_AMPLITUDE=1")
	}
	model := releasedPCMGenerationModel(t)
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 || len(pcm) < 80000 {
		t.Fatalf("JFK PCM rate=%d len=%d err=%v", rate, len(pcm), err)
	}
	// Eighty thousand samples bound the request while crossing the first
	// encoder chunk and exercising a true five-second PCM call.
	input := make([]float32, 80000)
	for i := range input {
		input[i] = pcm[i] * 3
	}
	before := append([]float32(nil), input...)
	type result struct {
		sub, tower []float32
		tokens     []int
		frames     []int64
	}
	run := func(backend string) result {
		t.Helper()
		s := &PCMGenerationStream{Model: model}
		var projector *DeviceSubsamplingProjector
		if backend != "simd" {
			projector = &DeviceSubsamplingProjector{Backend: backend}
			s.Projector = projector
			defer func() {
				if err := projector.Close(); err != nil {
					t.Fatal(err)
				}
			}()
		}
		var out result
		s.onStage = func(stage string, values []float32) {
			switch stage {
			case "subsampling":
				out.sub = append(out.sub, values...)
			case "tower":
				out.tower = append(out.tower, values...)
			}
		}
		part, frames, err := s.AppendPCM(context.Background(), input)
		if err != nil {
			t.Fatal(err)
		}
		out.tokens = append(out.tokens, part...)
		out.frames = append(out.frames, frames...)
		part, frames, err = s.Finish(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		out.tokens = append(out.tokens, part...)
		out.frames = append(out.frames, frames...)
		if projector != nil && projector.dispatches < 1 {
			t.Fatalf("%s projector did not dispatch", backend)
		}
		return out
	}
	ref := run("simd")
	if len(ref.sub) == 0 || len(ref.tower) == 0 || len(ref.tokens) == 0 {
		t.Fatal("empty amplified reference")
	}
	for _, backend := range []string{"ptx", "vulkan"} {
		t.Run(backend, func(t *testing.T) {
			var previous bool
			var statsBefore ptx.Stats
			if backend == "ptx" {
				previous = ptx.SetStatsEnabled(true)
				defer ptx.SetStatsEnabled(previous)
				statsBefore = ptx.StatsSnapshot()
			}
			got := run(backend)
			for _, stage := range []struct {
				name                string
				got, want           []float32
				abs, rel, meanLimit float64
			}{{"subsampling", got.sub, ref.sub, 3e-3, 4e-5, 1e-4}, {"tower", got.tower, ref.tower, 3e-4, 2e-5, 2e-5}} {
				if len(stage.got) != len(stage.want) {
					t.Fatalf("%s length got=%d want=%d", stage.name, len(stage.got), len(stage.want))
				}
				var max, sum float64
				var outside int
				for i, value := range stage.got {
					d := math.Abs(float64(value - stage.want[i]))
					max = math.Max(max, d)
					sum += d
					if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || d > stage.abs+stage.rel*math.Abs(float64(stage.want[i])) {
						outside++
					}
				}
				mean := sum / float64(len(stage.got))
				t.Logf("%s max_abs=%g mean_abs=%g outside=%d", stage.name, max, mean, outside)
				if outside != 0 || mean > stage.meanLimit {
					t.Fatalf("%s differs from SIMD", stage.name)
				}
			}
			if len(got.tokens) != len(ref.tokens) || len(got.frames) != len(ref.frames) {
				t.Fatal("decision length differs from SIMD")
			}
			for i := range got.tokens {
				if got.tokens[i] != ref.tokens[i] || got.frames[i] != ref.frames[i] {
					t.Fatalf("decision %d differs from SIMD", i)
				}
			}
			if backend == "ptx" {
				after := ptx.StatsSnapshot()
				if after.Mallocs-statsBefore.Mallocs != after.Frees-statsBefore.Frees || after.MallocBytes-statsBefore.MallocBytes != after.FreeBytes-statsBefore.FreeBytes {
					t.Fatalf("PTX resources leaked: before=%+v after=%+v", statsBefore, after)
				}
			}
		})
	}
	for i := range input {
		if input[i] != before[i] {
			t.Fatalf("caller PCM mutated at %d", i)
		}
	}
}
