package pockettts

import (
	"math"
	"os"
	"testing"
)

func releasedTrainingConfig(t *testing.T) Config {
	t.Helper()
	data, err := os.ReadFile("testdata/english-upstream.yaml")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseConfig(data)
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestProductionTrainingShapeAdmission(t *testing.T) {
	cfg := releasedTrainingConfig(t)
	shape := TrainingShape{MicroBatchRows: 1, TargetFrames: ProductionTrainingTargetFrames, VoiceFrames: ProductionTrainingVoiceFrames, TextTokens: 512, GradientAccumulation: 64, FlowBatchMultiplier: 1}
	limits := ProductionTrainingShapeLimits(64 << 30)
	plan, err := PlanTrainingShape(cfg, shape, limits)
	if err != nil {
		t.Fatal(err)
	}
	if plan.SequenceRows != 950 || plan.EffectiveBatchRows != 64 || plan.TransformerLayers != 6 || plan.ParameterElements != 89449730 || plan.ParameterBytes != plan.ParameterElements*4 || plan.GradientBytes != plan.ParameterBytes || plan.AdamMomentBytes != 2*plan.ParameterBytes || plan.EMABytes != plan.ParameterBytes || plan.ActivationUpperBytes <= 0 || plan.ResidentUpperBytes != plan.ParameterBytes*5+plan.ActivationUpperBytes || plan.ResidentUpperBytes > limits.MaxResidentBytes {
		t.Fatalf("plan=%+v", plan)
	}
	t.Logf("released production row: sequence=%d parameters=%.2f MiB activations<=%.2f MiB resident<=%.2f MiB", plan.SequenceRows, float64(plan.ParameterBytes)/(1<<20), float64(plan.ActivationUpperBytes)/(1<<20), float64(plan.ResidentUpperBytes)/(1<<20))
}

func TestTeacherProductionTrainingShapeAdmission(t *testing.T) {
	cfg := releasedTrainingConfig(t)
	cfg.FlowLM.Transformer.NumLayers = 24
	shape := TrainingShape{MicroBatchRows: 1, TargetFrames: 375, VoiceFrames: 62, TextTokens: 128, GradientAccumulation: 64, FlowBatchMultiplier: 1}
	plan, err := PlanTrainingShape(cfg, shape, ProductionTrainingShapeLimits(64<<30))
	if err != nil {
		t.Fatal(err)
	}
	if plan.TransformerLayers != 24 || plan.SequenceRows != 566 || plan.ParameterElements != 316015874 || plan.ResidentUpperBytes <= plan.ParameterBytes*5 {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestTrainingShapeRejectsBeforeAllocation(t *testing.T) {
	cfg := releasedTrainingConfig(t)
	valid := TrainingShape{MicroBatchRows: 1, TargetFrames: 375, VoiceFrames: 62, TextTokens: 128, GradientAccumulation: 64, FlowBatchMultiplier: 1}
	limits := ProductionTrainingShapeLimits(64 << 30)
	cases := []TrainingShape{{}, {MicroBatchRows: 2, TargetFrames: 1, TextTokens: 1, GradientAccumulation: 1, FlowBatchMultiplier: 1}, {MicroBatchRows: 1, TargetFrames: 376, TextTokens: 1, GradientAccumulation: 1, FlowBatchMultiplier: 1}, {MicroBatchRows: 1, TargetFrames: 1, VoiceFrames: 63, TextTokens: 1, GradientAccumulation: 1, FlowBatchMultiplier: 1}, {MicroBatchRows: 1, TargetFrames: 1, TextTokens: 513, GradientAccumulation: 1, FlowBatchMultiplier: 1}, {MicroBatchRows: 1, TargetFrames: 1, TextTokens: 1, GradientAccumulation: 1, FlowBatchMultiplier: 2}}
	for i, shape := range cases {
		if _, err := PlanTrainingShape(cfg, shape, limits); err == nil {
			t.Fatalf("case %d accepted", i)
		}
	}
	low := limits
	low.MaxResidentBytes = 1
	if _, err := PlanTrainingShape(cfg, valid, low); err == nil {
		t.Fatal("accepted resident budget underflow")
	}
	bad := cfg
	bad.FlowLM.Transformer.DModel = math.MaxInt
	if _, err := PlanTrainingShape(bad, valid, limits); err == nil {
		t.Fatal("accepted malformed maximum config")
	}
	bad = cfg
	bad.FlowLM.Transformer.DModel = math.MaxInt - 15
	bad.FlowLM.Transformer.NumHeads = 16
	bad.FlowLM.LookupTable.Dim = bad.FlowLM.Transformer.DModel
	if _, err := trainingParameterElements(bad); err {
		t.Fatal("overflowing parameter arithmetic accepted")
	}
	if _, ok := trainingActivationUpperElements(cfg, math.MaxInt64/2, 375, 62); ok {
		t.Fatal("overflowing activation arithmetic accepted")
	}
	if _, err := PlanTrainingShape(cfg, valid, TrainingShapeLimits{}); err == nil {
		t.Fatal("accepted empty limits")
	}
}

func TestAdmittedTrainingWorkspaceRejectsForgeryAndMismatch(t *testing.T) {
	lm, flow := tinyFlowLMTraining(), tinyTrainableFlowHead()
	flow.Condition = tinyLinear(4, 4, -.02)
	for i := range flow.Time {
		flow.Time[i].Frequencies = make([]float32, 128)
		for j := range flow.Time[i].Frequencies {
			flow.Time[i].Frequencies[j] = float32(j + 1)
		}
		flow.Time[i].FC1 = tinyLinear(4, 256, .01)
	}
	weighting := &LSDWeightMLP{Layers: []LinearF32{{Weight: make([]float32, 64), Bias: make([]float32, 32), In: 2, Out: 32}, {Weight: make([]float32, 1024), Bias: make([]float32, 32), In: 32, Out: 32}, {Weight: make([]float32, 1024), Bias: make([]float32, 32), In: 32, Out: 32}, {Weight: make([]float32, 32), Bias: make([]float32, 1), In: 32, Out: 1}}}
	batch := tinyFlowLMBatch()
	cfg := releasedTrainingConfig(t)
	cfg.FlowLM.Transformer.DModel = 4
	cfg.FlowLM.Transformer.NumHeads = 2
	cfg.FlowLM.Transformer.NumLayers = 1
	cfg.FlowLM.Transformer.DimFeedforward = 6
	cfg.FlowLM.Transformer.LayerScale = .01
	cfg.FlowLM.Flow.Dim = 4
	cfg.FlowLM.Flow.Depth = 2
	cfg.FlowLM.LookupTable.Dim = 4
	cfg.FlowLM.LookupTable.NBins = 5
	cfg.Mimi.InnerDim = 2
	cfg.Mimi.Quantizer.Dimension = 2
	shape := TrainingShape{MicroBatchRows: 1, TargetFrames: batch.Frames, VoiceFrames: batch.VoiceFrames, TextTokens: len(batch.TextTokens), GradientAccumulation: 1, FlowBatchMultiplier: 1}
	plan, err := PlanTrainingShape(cfg, shape, ProductionTrainingShapeLimits(64<<30))
	if err != nil {
		t.Fatal(err)
	}
	if _, err = NewAdmittedTrainingWorkspace(lm, flow, weighting, batch, plan); err != nil {
		t.Fatal(err)
	}
	mutatedReport := plan
	mutatedReport.TargetFrames++
	mutatedReport.Hidden++
	if _, err = NewAdmittedTrainingWorkspace(lm, flow, weighting, batch, mutatedReport); err != nil {
		t.Fatal("mutable report fields changed private admission")
	}
	for _, bad := range []TrainingShapePlan{{}, func() TrainingShapePlan { x := plan; x.admission.targetFrames++; return x }(), func() TrainingShapePlan { x := plan; x.admission.hidden++; return x }(), func() TrainingShapePlan { x := plan; x.admission.flowDepth++; return x }()} {
		if _, err = NewAdmittedTrainingWorkspace(lm, flow, weighting, batch, bad); err == nil {
			t.Fatal("accepted forged or mismatched plan")
		}
	}
	larger := *lm
	larger.Embedding = append(append([]float32(nil), lm.Embedding...), make([]float32, lm.Hidden)...)
	larger.Vocabulary++
	if _, err = NewAdmittedTrainingWorkspace(&larger, flow, weighting, batch, plan); err == nil {
		t.Fatal("accepted larger embedding topology")
	}
	largerWeighting := *weighting
	largerWeighting.Layers = append(append([]LinearF32(nil), weighting.Layers...), LinearF32{Weight: []float32{0}, Bias: []float32{0}, In: 1, Out: 1})
	if _, err = NewAdmittedTrainingWorkspace(lm, flow, &largerWeighting, batch, plan); err == nil {
		t.Fatal("accepted larger weighting topology")
	}
	bf16 := *lm
	bf16.Input = lm.Input
	bf16.Input.WeightBF16 = []uint16{1}
	if _, err = NewAdmittedTrainingWorkspace(&bf16, flow, weighting, batch, plan); err == nil {
		t.Fatal("accepted BF16 side storage")
	}
	biased := *lm
	biased.Input = lm.Input
	biased.Input.Bias = []float32{0, 0, 0, 0}
	if _, err = NewAdmittedTrainingWorkspace(&biased, flow, weighting, batch, plan); err == nil {
		t.Fatal("accepted forbidden training bias")
	}
	reshaped := *lm
	reshaped.BOS = append(append([]float32(nil), lm.BOS...), 0)
	reshaped.BOSBeforeVoice = append([]float32(nil), lm.BOSBeforeVoice[:len(lm.BOSBeforeVoice)-1]...)
	if _, err = NewAdmittedTrainingWorkspace(&reshaped, flow, weighting, batch, plan); err == nil {
		t.Fatal("accepted reshaped BOS parameters with equal total elements")
	}
}

func TestProductionTrainingFrameConstants(t *testing.T) {
	if ProductionTrainingTargetFrames != 375 || ProductionTrainingVoiceFrames != 62 {
		t.Fatalf("frames target=%d voice=%d", ProductionTrainingTargetFrames, ProductionTrainingVoiceFrames)
	}
}
