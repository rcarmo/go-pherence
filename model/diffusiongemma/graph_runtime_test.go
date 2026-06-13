package diffusiongemma

import (
	"math/rand"
	"testing"
)

type graphRuntimeDenoiser struct {
	calls          int
	prevTempSeen   float64
	prevLogitsSeen bool
	graphSeen      ExecutionGraph
}

func (d *graphRuntimeDenoiser) Denoise(in ForwardInput) (ForwardOutput, error) {
	d.calls++
	if d.calls == 2 {
		d.prevTempSeen = in.SelfConditioningTemperature
		d.prevLogitsSeen = len(in.SelfConditioningLogits) == len(in.Canvas) && len(in.SelfConditioningLogits[0]) == 4
		d.graphSeen = in.Graph
	}
	logits := make([][]float32, len(in.Canvas))
	for i := range logits {
		row := make([]float32, 4)
		row[(d.calls+i)%4] = 3
		logits[i] = row
	}
	return ForwardOutput{Logits: logits}, nil
}

func TestGenerateCanvasPassesPreviousRawLogitsAndTemperatureToGraph(t *testing.T) {
	d := &graphRuntimeDenoiser{}
	cfg := DefaultDenoisingConfig()
	cfg.MaxDenoisingSteps = 2
	cfg.StabilityThreshold = 0
	cfg.ConfidenceThreshold = 0
	cfg.Sampler.Mode = SamplerModeArgmax
	_, err := GenerateCanvas(d, []int{1, 2, 3}, cfg, 2, 4, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if d.calls != 2 {
		t.Fatalf("calls=%d want 2", d.calls)
	}
	firstStepTemp := LinearTemperature(cfg.TMin, cfg.TMax, cfg.MaxDenoisingSteps, 2)
	if d.prevTempSeen != firstStepTemp {
		t.Fatalf("previous SC temperature = %.6f, want %.6f", d.prevTempSeen, firstStepTemp)
	}
	if !d.prevLogitsSeen {
		t.Fatalf("second call did not receive previous raw logits")
	}
	if d.graphSeen.Phase != ExecutionGraphDecode || d.graphSeen.PromptLength != 3 || d.graphSeen.CanvasLength != 2 {
		t.Fatalf("graph = %+v, want decode P=3 C=2", d.graphSeen)
	}
}
