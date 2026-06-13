package diffusiongemma

import (
	"math/rand"
	"testing"
)

type snapshotTestDenoiser struct {
	vocabSize int
}

func (d snapshotTestDenoiser) Denoise(in ForwardInput) (ForwardOutput, error) {
	logits := make([][]float32, len(in.Canvas))
	token := in.Step
	if token >= d.vocabSize {
		token = d.vocabSize - 1
	}
	for i := range logits {
		row := make([]float32, d.vocabSize)
		row[token] = 10
		logits[i] = row
	}
	return ForwardOutput{Logits: logits}, nil
}

func TestGenerateCanvasStepCallbackSnapshotsAreCopies(t *testing.T) {
	const canvasLength = 3
	const vocabSize = 8
	var snapshots []DiffusionStepSnapshot
	cfg := DefaultDenoisingConfig()
	cfg.MaxDenoisingSteps = 2
	cfg.StabilityThreshold = 0
	cfg.ConfidenceThreshold = 0
	cfg.StepCallback = func(snapshot DiffusionStepSnapshot) error {
		if len(snapshot.Canvas) != canvasLength {
			t.Fatalf("snapshot canvas length = %d, want %d", len(snapshot.Canvas), canvasLength)
		}
		if len(snapshot.AcceptedMask) != canvasLength {
			t.Fatalf("snapshot accepted mask length = %d, want %d", len(snapshot.AcceptedMask), canvasLength)
		}
		stored := DiffusionStepSnapshot{
			Step:         snapshot.Step,
			Temperature:  snapshot.Temperature,
			Canvas:       append([]int(nil), snapshot.Canvas...),
			AcceptedMask: append([]bool(nil), snapshot.AcceptedMask...),
			MeanEntropy:  snapshot.MeanEntropy,
			Stopped:      snapshot.Stopped,
		}
		snapshots = append(snapshots, stored)
		// Deliberately mutate the callback-owned slices. This must not affect the
		// runtime's canvas or later snapshots.
		snapshot.Canvas[0] = 99
		snapshot.AcceptedMask[0] = false
		return nil
	}
	res, err := GenerateCanvas(snapshotTestDenoiser{vocabSize: vocabSize}, []int{1, 2}, cfg, canvasLength, vocabSize, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(snapshots), 2; got != want {
		t.Fatalf("snapshot count = %d, want %d", got, want)
	}
	for _, tok := range snapshots[0].Canvas {
		if tok != 2 {
			t.Fatalf("first snapshot canvas = %v, want all 2", snapshots[0].Canvas)
		}
	}
	for _, tok := range snapshots[1].Canvas {
		if tok != 1 {
			t.Fatalf("second snapshot canvas = %v, want all 1", snapshots[1].Canvas)
		}
	}
	for _, accepted := range snapshots[1].AcceptedMask {
		if !accepted {
			t.Fatalf("accepted mask = %v, want all true", snapshots[1].AcceptedMask)
		}
	}
	for _, tok := range res.Canvas {
		if tok != 1 {
			t.Fatalf("final canvas = %v, want all 1", res.Canvas)
		}
	}
}
