package diffusiongemma

import (
	"math/rand"
	"testing"
)

type entropyModeDenoiser struct{}

func (entropyModeDenoiser) Denoise(in ForwardInput) (ForwardOutput, error) {
	logits := make([][]float32, len(in.Canvas))
	for pos := range logits {
		row := make([]float32, 4)
		if pos == 0 {
			row[1] = 2.0
		} else {
			// High-entropy rows should not all be accepted when the entropy bound
			// is tighter than the previous accepted entropy.
			row[0], row[1], row[2], row[3] = 0.1, 0.2, 0.3, 0.4
		}
		logits[pos] = row
	}
	return ForwardOutput{Logits: logits}, nil
}

func TestGenerateCanvasEntropyBoundModePartialAcceptsAndRenoises(t *testing.T) {
	cfg := DefaultDenoisingConfig()
	cfg.MaxDenoisingSteps = 1
	cfg.StabilityThreshold = 0
	cfg.ConfidenceThreshold = 0
	cfg.Sampler.Mode = SamplerModeEntropyBound
	cfg.Sampler.EntropyBound = 0.1
	var snapshot DiffusionStepSnapshot
	cfg.StepCallback = func(s DiffusionStepSnapshot) error {
		snapshot = s
		return nil
	}
	res, err := GenerateCanvas(entropyModeDenoiser{}, []int{1, 2}, cfg, 4, 4, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if got := len(res.Steps); got != 1 {
		t.Fatalf("steps = %d, want 1", got)
	}
	if accepted := res.Steps[0].Accepted; accepted <= 0 || accepted >= 4 {
		t.Fatalf("accepted = %d, want partial acceptance", accepted)
	}
	if len(snapshot.AcceptedMask) != 4 {
		t.Fatalf("accepted mask length = %d", len(snapshot.AcceptedMask))
	}
	seenAccepted, seenRejected := false, false
	for _, accepted := range snapshot.AcceptedMask {
		seenAccepted = seenAccepted || accepted
		seenRejected = seenRejected || !accepted
	}
	if !seenAccepted || !seenRejected {
		t.Fatalf("accepted mask = %v, want accepted and rejected positions", snapshot.AcceptedMask)
	}
	wantArgmax := []int{1, 3, 3, 3}
	for i, want := range wantArgmax {
		if snapshot.Canvas[i] != want {
			t.Fatalf("snapshot canvas = %v, want argmax canvas %v", snapshot.Canvas, wantArgmax)
		}
		if res.Canvas[i] != want {
			t.Fatalf("final canvas = %v, want argmax canvas %v", res.Canvas, wantArgmax)
		}
	}
}
