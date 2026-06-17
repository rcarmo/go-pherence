package diffusiongemma

import (
	"math"
	"math/rand"
	"testing"
)

type recordingDenoiser struct {
	vocab    int
	tempInvs []float32
}

func (d *recordingDenoiser) Denoise(in ForwardInput) (ForwardOutput, error) {
	d.tempInvs = append(d.tempInvs, in.SCTempInv)
	logits := make([][]float32, len(in.Canvas))
	for i := range logits {
		row := make([]float32, d.vocab)
		// Keep argmax stable but entropy non-zero so the loop does not stop when
		// ConfidenceThreshold is disabled.
		row[i%d.vocab] = 1
		logits[i] = row
	}
	return ForwardOutput{Logits: logits, SelfConditioning: []float32{1}}, nil
}

func TestGenerateCanvasPassesPreviousStepTempInvForRawSelfConditioning(t *testing.T) {
	den := &recordingDenoiser{vocab: 5}
	cfg := DefaultDenoisingConfig()
	cfg.MaxDenoisingSteps = 3
	cfg.TMin = 0.4
	cfg.TMax = 0.8
	cfg.StabilityThreshold = 100
	cfg.ConfidenceThreshold = 0

	_, err := GenerateCanvas(den, []int{1, 2}, cfg, 2, den.vocab, rand.New(rand.NewSource(7)))
	if err != nil {
		t.Fatal(err)
	}
	if len(den.tempInvs) != cfg.MaxDenoisingSteps {
		t.Fatalf("recorded %d temp invs, want %d", len(den.tempInvs), cfg.MaxDenoisingSteps)
	}
	want := []float32{
		1.0,
		float32(1.0 / LinearTemperature(cfg.TMin, cfg.TMax, cfg.MaxDenoisingSteps, 3)),
		float32(1.0 / LinearTemperature(cfg.TMin, cfg.TMax, cfg.MaxDenoisingSteps, 2)),
	}
	for i := range want {
		if math.Abs(float64(den.tempInvs[i]-want[i])) > 1e-6 {
			t.Fatalf("tempInv[%d]=%.8f want %.8f (all=%v)", i, den.tempInvs[i], want[i], den.tempInvs)
		}
	}
}

func TestGenerateCanvasZeroConfigUsesReferenceDefaults(t *testing.T) {
	den := &recordingDenoiser{vocab: 5}
	res, err := GenerateCanvas(den, []int{1, 2}, DenoisingConfig{}, 2, den.vocab, NewMT19937RNG(0))
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Steps) == 0 {
		t.Fatal("no denoising steps recorded")
	}
	defaults := DefaultDenoisingConfig()
	if res.Steps[0].Temperature != defaults.TMax {
		t.Fatalf("first temperature=%g want default t_max=%g", res.Steps[0].Temperature, defaults.TMax)
	}
}

func TestRenoiseCanvasWithPredrawnTokens(t *testing.T) {
	got := RenoiseCanvasWithTokens([]int{10, 11, 12, 13}, []bool{false, true, false, true}, []int{1, 2, 3, 4})
	want := []int{1, 11, 3, 13}
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v want %v", got, want)
		}
	}
}

func TestAcceptCanvasAllowsZeroEntropyBound(t *testing.T) {
	got := AcceptCanvas([]int{0, 0, 0}, []int{10, 11, 12}, []float64{0.3, 0.1, 0.2}, 0)
	wantMask := []bool{false, true, false}
	if got.Accepted != 1 {
		t.Fatalf("accepted=%d want 1 (mask=%v canvas=%v)", got.Accepted, got.AcceptedMask, got.Canvas)
	}
	for i := range wantMask {
		if got.AcceptedMask[i] != wantMask[i] {
			t.Fatalf("mask=%v want %v", got.AcceptedMask, wantMask)
		}
	}
	if got.Canvas[1] != 11 || got.Canvas[0] != 0 || got.Canvas[2] != 0 {
		t.Fatalf("canvas=%v", got.Canvas)
	}
}

type seedCanvasDenoiser struct{ seen []int }

func (d *seedCanvasDenoiser) Denoise(in ForwardInput) (ForwardOutput, error) {
	if len(d.seen) == 0 {
		d.seen = append([]int(nil), in.Canvas...)
	}
	logits := make([][]float32, len(in.Canvas))
	for i := range logits {
		logits[i] = []float32{1, 0, 0, 0, 0}
	}
	return ForwardOutput{Logits: logits}, nil
}

type precomputedSamplerDenoiser struct{}

func (d precomputedSamplerDenoiser) Denoise(in ForwardInput) (ForwardOutput, error) {
	return ForwardOutput{
		ArgmaxCanvas:     []int{3, 4},
		SampledCanvas:    []int{1, 2},
		Entropy:          []float64{0.01, 0.02},
		SelfConditioning: []float32{1, 2},
	}, nil
}

func TestGenerateCanvasUsesPrecomputedSamplerOutputs(t *testing.T) {
	cfg := DenoisingConfig{MaxDenoisingSteps: 1, TMin: 1, TMax: 1, StabilityThreshold: 99, ConfidenceThreshold: 0, Sampler: SamplerConfig{EntropyBound: 0}}
	res, err := GenerateCanvas(precomputedSamplerDenoiser{}, nil, cfg, 2, 8, NewMT19937RNG(0))
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Canvas) != 2 || res.Canvas[0] != 3 || res.Canvas[1] != 4 {
		t.Fatalf("canvas=%v want argmax [3 4]", res.Canvas)
	}
	if len(res.Steps) != 1 || res.Steps[0].MeanEntropy != 0.015 {
		t.Fatalf("steps=%+v", res.Steps)
	}
	if res.Steps[0].FirstArgmax != 3 || res.Steps[0].FirstSampled != 1 || res.Steps[0].FirstEntropy != 0.01 || !res.Steps[0].FirstAccepted {
		t.Fatalf("first-position diagnostics=%+v", res.Steps[0])
	}
	if res.Steps[0].MaxEntropy != 0.02 || res.Steps[0].MaxEntropyPos != 1 {
		t.Fatalf("max-entropy diagnostics=%+v", res.Steps[0])
	}
}

func TestTopLogitsReturnsDescendingIDsAndValues(t *testing.T) {
	ids, vals := topLogits([]float32{0.5, 2.0, -1.0, 2.5, 1.5}, 3)
	wantIDs := []int{3, 1, 4}
	wantVals := []float32{2.5, 2.0, 1.5}
	if len(ids) != len(wantIDs) || len(vals) != len(wantVals) {
		t.Fatalf("topLogits len ids=%v vals=%v", ids, vals)
	}
	for i := range wantIDs {
		if ids[i] != wantIDs[i] || vals[i] != wantVals[i] {
			t.Fatalf("topLogits=%v/%v want %v/%v", ids, vals, wantIDs, wantVals)
		}
	}
}

func TestGenerateCanvasRecordsEntropyProbes(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_ENTROPY_PROBE_POSITIONS", "1,0,1,bad,9")
	cfg := DenoisingConfig{MaxDenoisingSteps: 1, TMin: 1, TMax: 1, StabilityThreshold: 99, ConfidenceThreshold: 0, Sampler: SamplerConfig{EntropyBound: 0}}
	res, err := GenerateCanvas(precomputedSamplerDenoiser{}, nil, cfg, 2, 8, NewMT19937RNG(0))
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Steps) != 1 || len(res.Steps[0].EntropyProbes) != 2 {
		t.Fatalf("entropy probes=%+v", res.Steps)
	}
	p0 := res.Steps[0].EntropyProbes[0]
	if p0.Position != 1 || p0.Argmax != 4 || p0.Sampled != 2 || p0.Entropy != 0.02 || p0.Accepted || len(p0.TopIDs) != 0 || len(p0.TopLogits) != 0 {
		t.Fatalf("probe[0]=%+v", p0)
	}
	p1 := res.Steps[0].EntropyProbes[1]
	if p1.Position != 0 || p1.Argmax != 3 || p1.Sampled != 1 || p1.Entropy != 0.01 || !p1.Accepted || len(p1.TopIDs) != 0 || len(p1.TopLogits) != 0 {
		t.Fatalf("probe[1]=%+v", p1)
	}
}

func TestGenerateTokenIDsPreservesSeedZero(t *testing.T) {
	mk := func() (*Engine, *seedCanvasDenoiser) {
		den := &seedCanvasDenoiser{}
		eng := &Engine{Model: &Model{Shape: Shape{CanvasLength: 3, VocabSize: 5}, Denoising: DenoisingConfig{MaxDenoisingSteps: 1, TMin: 1, TMax: 1, StabilityThreshold: 99, ConfidenceThreshold: 0, Sampler: SamplerConfig{EntropyBound: 0}}}, Denoiser: den}
		return eng, den
	}
	eng0, den0 := mk()
	if _, err := eng0.GenerateTokenIDs(nil, InferenceOptions{MaxNewTokens: 1, CanvasLength: 3, Seed: 0}); err != nil {
		t.Fatal(err)
	}
	eng1, den1 := mk()
	if _, err := eng1.GenerateTokenIDs(nil, InferenceOptions{MaxNewTokens: 1, CanvasLength: 3, Seed: 1}); err != nil {
		t.Fatal(err)
	}
	if len(den0.seen) != 3 || len(den1.seen) != 3 {
		t.Fatalf("missing initial canvases seed0=%v seed1=%v", den0.seen, den1.seen)
	}
	same := true
	for i := range den0.seen {
		if den0.seen[i] != den1.seen[i] {
			same = false
		}
	}
	if same {
		t.Fatalf("seed 0 appears remapped to seed 1: %v", den0.seen)
	}
}

func TestGenerateTokenIDsRejectsOversizedCanvasOverride(t *testing.T) {
	eng := &Engine{Model: &Model{Shape: Shape{CanvasLength: 3, VocabSize: 5}, Denoising: DefaultDenoisingConfig()}, Denoiser: &seedCanvasDenoiser{}}
	if _, err := eng.GenerateTokenIDs(nil, InferenceOptions{MaxNewTokens: 1, CanvasLength: 4}); err == nil {
		t.Fatal("GenerateTokenIDs accepted canvas override larger than model canvas_length")
	}
}

func TestDenoisingConfigFromDefaultsPreservesZeroControls(t *testing.T) {
	cfg := DenoisingConfigFromDefaults(GenerationDefaults{
		MaxDenoisingSteps:   48,
		TMin:                0.4,
		TMax:                0.8,
		StabilityThreshold:  0,
		ConfidenceThreshold: 0,
		EntropyBound:        0,
	})
	if cfg.StabilityThreshold != 0 || cfg.ConfidenceThreshold != 0 || cfg.Sampler.EntropyBound != 0 {
		t.Fatalf("zero controls not preserved: stability=%d confidence=%g entropy=%g", cfg.StabilityThreshold, cfg.ConfidenceThreshold, cfg.Sampler.EntropyBound)
	}
}
