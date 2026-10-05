package pockettts

import "testing"

func sharedZeroWorkspaceFixture(t *testing.T) (*TrainingWorkspace, *FlowLMTrainingCPU, *FlowHeadCPU, *LSDWeightMLP, *ResidentTrainingBackend) {
	t.Helper()
	lm, flow, w := tinyFlowLMTraining(), tinyTrainableFlowHead(), tinyLSDWeighting()
	flow.Condition = tinyLinear(4, 4, -.02)
	b := &ResidentTrainingBackend{Scope: "vulkan-resident-sharedworkspace-hybrid-v1"}
	flow.TrainingBackend = b
	flow.LossBackend = b
	flow.BranchBackend = b
	flow.StepReductionBackend = b
	lm.TransformerBackend = b
	lm.ProjectionBackend = b
	lm.SequenceBackend = b
	lm.EOSBackend = b
	w.TrainingBackend = b
	ws, e := newTrainingWorkspace(lm, flow, tinyFlowLMBatch(), true)
	if e != nil {
		t.Fatal(e)
	}
	return ws, lm, flow, w, b
}
func TestTrainingSharedZeroWorkspaceOwnership(t *testing.T) {
	ws, lm, flow, w, b := sharedZeroWorkspaceFixture(t)
	if !ws.compatible(lm, flow) || ws.zeroGradientOwner != b || !flowLMGradientMatches(ws.flowLMGradients, lm) {
		t.Fatal("pooled topology")
	}
	params := map[string][]float32{}
	addFlowLMGradients(params, ws.flowLMGradients)
	addFlowHeadGradientParameters(params, ws.flowGradients)
	unique := map[*float32]int{}
	logical := 0
	for _, v := range params {
		logical += len(v)
		if len(v) > 0 {
			unique[&v[0]] = len(v)
		}
		if !zeroTrainingSeed(v) {
			t.Fatal("nonzero placeholder")
		}
	}
	elements := 0
	for _, n := range unique {
		elements += n
	}
	if elements >= logical {
		t.Fatal("zero backing did not pool")
	}
	ws.reset()
	if !ws.compatible(lm, flow) {
		t.Fatal("reset changed zeros")
	}
	// Returned placeholders cannot hide mutation by clearing on the next step.
	ws.flowGradients.Input.Weight[0] = 1
	if ws.compatible(lm, flow) {
		t.Fatal("host gradient mutation accepted")
	}
	if _, _, e := PocketTrainingStepInto(lm, flow, w, tinyFlowLMBatch(), tinyTrainingStepSamples(), DefaultTrainingStepConfig(), ws); e == nil {
		t.Fatal("mutation reached arithmetic")
	}
}
func TestTrainingSharedZeroWorkspaceChangedOwner(t *testing.T) {
	for _, which := range []string{"flow", "loss", "branch", "step", "transformer", "projection", "sequence", "eos", "attention", "weight"} {
		t.Run(which, func(t *testing.T) {
			ws, lm, flow, w, _ := sharedZeroWorkspaceFixture(t)
			switch which {
			case "flow":
				flow.TrainingBackend = nil
			case "loss":
				flow.LossBackend = nil
			case "branch":
				flow.BranchBackend = nil
			case "step":
				flow.StepReductionBackend = nil
			case "transformer":
				lm.TransformerBackend = nil
			case "projection":
				lm.ProjectionBackend = nil
			case "sequence":
				lm.SequenceBackend = nil
			case "eos":
				lm.EOSBackend = nil
			case "attention":
				lm.AttentionBackwardBackend = &namedAttentionBackend{}
			case "weight":
				w.TrainingBackend = nil
			}
			if which != "weight" && ws.compatible(lm, flow) {
				t.Fatal("changed owner compatible")
			}
			if _, _, e := PocketTrainingStepInto(lm, flow, w, tinyFlowLMBatch(), tinyTrainingStepSamples(), DefaultTrainingStepConfig(), ws); e == nil {
				t.Fatal("changed owner admitted")
			}
		})
	}
}
func TestTrainingPublicWorkspaceGradientsRemainIndependent(t *testing.T) {
	_, lm, flow, _, _ := sharedZeroWorkspaceFixture(t)
	ws, e := NewTrainingWorkspace(lm, flow, tinyFlowLMBatch())
	if e != nil {
		t.Fatal(e)
	}
	if ws.zeroGradientOwner != nil {
		t.Fatal("public workspace pooled")
	}
	ws.flowGradients.Time[0].FC1.Weight[0] = 1
	if ws.flowGradients.Time[1].FC1.Weight[0] != 0 {
		t.Fatal("CPU gradients alias")
	}
	ws.reset()
	if ws.flowGradients.Time[0].FC1.Weight[0] != 0 {
		t.Fatal("CPU reset")
	}
}

func TestTrainingSharedForwardWorkspacePlaceholders(t *testing.T) {
	ws, lm, _, _, owner := sharedZeroWorkspaceFixture(t)
	batch := tinyFlowLMBatch()
	mask := tinyTrainingStepSamples().Mask
	owner.forwardZeroWorkspace = ws
	before := len(*ws.zeroGradientPool)
	a, err := lm.buildEOSForwardTrainingTape(batch, mask, .1, 1<<24)
	if err != nil {
		t.Fatal(err)
	}
	second, err := lm.buildEOSForwardTrainingTape(batch, mask, .1, 1<<24)
	if err != nil {
		t.Fatal(err)
	}
	distinct := map[int]bool{}
	for name, binding := range a.ResidentBindings {
		x := a.Tape.Tensors[binding.Gradient].Values
		n := len(x)
		pooled, ok := (*ws.zeroGradientPool)[n]
		if distinct[binding.Gradient] {
			t.Fatal("merged gradient root IDs", name)
		}
		distinct[binding.Gradient] = true
		if ok && (&x[0] != &pooled[0] || &second.Tape.Tensors[second.ResidentBindings[name].Gradient].Values[0] != &pooled[0]) {
			t.Fatal("existing extent not reused", name)
		}
	}
	if len(*ws.zeroGradientPool) != before {
		t.Fatal("retained pool grew with graph scratch")
	}
	owner.forwardZeroWorkspace = nil
	local, err := lm.buildEOSForwardTrainingTape(batch, mask, .1, 1<<24)
	if err != nil {
		t.Fatal(err)
	}
	for name, binding := range a.ResidentBindings {
		if f32SlicesOverlap(a.Tape.Tensors[binding.Gradient].Values, local.Tape.Tensors[local.ResidentBindings[name].Gradient].Values) {
			t.Fatal("unscoped builder borrowed workspace", name)
		}
	}
	owner.forwardZeroWorkspace = ws
	ws.flowGradients.Input.Weight[0] = 1
	if _, err = lm.buildEOSForwardTrainingTape(batch, mask, .1, 1<<24); err == nil {
		t.Fatal("mutated pooled source admitted")
	}
}

func TestTrainingSharedForwardWorkspaceProvenance(t *testing.T) {
	for _, kind := range []string{"owner", "shape", "model"} {
		t.Run(kind, func(t *testing.T) {
			ws, lm, _, _, owner := sharedZeroWorkspaceFixture(t)
			owner.forwardZeroWorkspace = ws
			switch kind {
			case "owner":
				ws.zeroGradientOwner = &ResidentTrainingBackend{Scope: owner.Scope}
			case "shape":
				ws.frames++
			case "model":
				ws.flowLM = tinyFlowLMTraining()
			}
			if _, err := lm.buildEOSForwardTrainingTape(tinyFlowLMBatch(), tinyTrainingStepSamples().Mask, .1, 1<<24); err == nil {
				t.Fatal("foreign scoped workspace admitted", kind)
			}
		})
	}
}

func TestTrainingSharedForwardWorkspaceStepCleanup(t *testing.T) {
	ws, lm, flow, w, owner := sharedZeroWorkspaceFixture(t)
	if _, _, err := PocketTrainingStepInto(lm, flow, w, tinyFlowLMBatch(), tinyTrainingStepSamples(), DefaultTrainingStepConfig(), ws); err == nil {
		t.Fatal("inactive backend accepted")
	}
	if owner.forwardZeroWorkspace != nil {
		t.Fatal("step exit retained workspace borrow")
	}
	owner.forwardZeroWorkspace = ws
	if _, _, err := PocketTrainingStepInto(lm, flow, w, tinyFlowLMBatch(), tinyTrainingStepSamples(), DefaultTrainingStepConfig(), ws); err == nil {
		t.Fatal("nested workspace borrow")
	}
	if owner.forwardZeroWorkspace != ws {
		t.Fatal("nested rejection cleared earlier borrow")
	}
}

func TestTrainingSharedStageWorkspacePlaceholders(t *testing.T) {
	ws, lm, _, _, _ := sharedZeroWorkspaceFixture(t)
	rows := 3
	sequence, seed := make([]float32, rows*lm.Hidden), make([]float32, rows*lm.Hidden)
	before := len(*ws.zeroGradientPool)
	stage, err := buildTransformerLayerStage(lm.Transformer, 0, sequence, seed, rows, true, 1<<24, ws.zeroGradientPool)
	if err != nil {
		t.Fatal(err)
	}
	local, err := buildTransformerLayerStage(lm.Transformer, 0, sequence, seed, rows, true, 1<<24)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[int]bool{}
	for name, binding := range stage.ResidentBindings {
		values := stage.Tape.Tensors[binding.Gradient].Values
		pooled, ok := (*ws.zeroGradientPool)[len(values)]
		if seen[binding.Gradient] {
			t.Fatal("stage placeholder IDs merged")
		}
		seen[binding.Gradient] = true
		if ok && &values[0] != &pooled[0] {
			t.Fatal("stage pool extent copied", name)
		}
		if f32SlicesOverlap(values, local.Tape.Tensors[local.ResidentBindings[name].Gradient].Values) {
			t.Fatal("unscoped stage borrowed pool", name)
		}
	}
	if len(*ws.zeroGradientPool) != before {
		t.Fatal("stage added scratch extents to workspace")
	}
	if lm.Transformer.FinalWeight != nil {
		if _, err = buildTransformerFinalNormStage(lm.Transformer, sequence, seed, rows, true, 1<<24, ws.zeroGradientPool); err != nil {
			t.Fatal(err)
		}
	}
	for _, values := range *ws.zeroGradientPool {
		if len(values) == len(lm.Transformer.Layers[0].FC1.Weight) {
			values[0] = 1
			break
		}
	}
	if _, err = buildTransformerLayerStage(lm.Transformer, 0, sequence, seed, rows, true, 1<<24, ws.zeroGradientPool); err == nil {
		t.Fatal("mutated stage placeholder admitted")
	}
}
