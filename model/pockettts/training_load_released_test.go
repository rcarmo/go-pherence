package pockettts

import (
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func releasedTrainingModels(tb testing.TB) (*FlowLMTrainingCPU, *FlowHeadCPU, *LSDWeightMLP, Config) {
	tb.Helper()
	path := releasedAsset(tb, "GO_PHERENCE_POCKETTTS_VOICE_MODEL", "fb0dc01b0d4d2e1c905b7a3e0676e3d9c96d5ae460e24e3ab94981805babf997", 219029196)
	file, err := safetensors.Open(path)
	if err != nil {
		tb.Fatal(err)
	}
	defer file.Close()
	cfg := releasedConfig(tb)
	lm, err := LoadFlowLMTrainingCPU(file, cfg)
	if err != nil {
		tb.Fatal(err)
	}
	flow, err := LoadFlowHeadTrainingCPU(file, cfg)
	if err != nil {
		tb.Fatal(err)
	}
	return lm, flow, NewDefaultLSDWeightMLP(), cfg
}

func TestReleasedTrainingLoadOwnedF32(t *testing.T) {
	lm, flow, weighting, cfg := releasedTrainingModels(t)
	if lm.Hidden != cfg.FlowLM.Transformer.DModel || lm.LatentDim != cfg.Mimi.InnerDim || lm.Vocabulary != cfg.FlowLM.LookupTable.NBins+1 || len(weighting.Layers) != 4 {
		t.Fatal("released training topology mismatch")
	}
	if len(lm.Input.WeightBF16) != 0 || len(lm.Input.Weight) != lm.Input.In*lm.Input.Out || len(flow.Input.WeightBF16) != 0 || len(flow.Input.Weight) != flow.Input.In*flow.Input.Out {
		t.Fatal("released training linears do not own F32 weights")
	}
	for _, layer := range lm.Transformer.Layers {
		for _, linear := range []LinearF32{layer.InProjection, layer.OutProjection, layer.FC1, layer.FC2} {
			if len(linear.WeightBF16) != 0 || len(linear.Weight) != linear.In*linear.Out {
				t.Fatal("released transformer linear is not owned F32")
			}
		}
	}
	for _, layer := range weighting.Layers {
		for _, value := range append(append([]float32(nil), layer.Weight...), layer.Bias...) {
			if value != 0 {
				t.Fatal("default LSD weighting is not zero-initialised")
			}
		}
	}
	params, err := fullParameterMap(lm, flow, weighting)
	if err != nil {
		t.Fatal(err)
	}
	var elements int64
	for _, values := range params {
		elements += int64(len(values))
	}
	want, ok := trainingParameterElements(cfg)
	if !ok || elements != want {
		t.Fatalf("released parameter elements=%d want=%d", elements, want)
	}
	path := os.Getenv("GO_PHERENCE_POCKETTTS_VOICE_MODEL")
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	for name, mutate := range map[string]func(*Config){
		"flow dimension": func(c *Config) { c.FlowLM.Flow.Dim++ },
		"context":        func(c *Config) { c.FlowLM.Transformer.Context = 1 },
		"RoPE period":    func(c *Config) { c.FlowLM.Transformer.MaxPeriod = 9999 },
	} {
		bad := cfg
		mutate(&bad)
		if _, err = LoadFlowLMTrainingCPU(file, bad); err == nil {
			t.Fatalf("accepted unsupported released training %s", name)
		}
	}
}

func TestReleasedProductionTrainingSoak(t *testing.T) {
	value := os.Getenv("GO_PHERENCE_POCKETTTS_RELEASED_TRAINING_STEPS")
	if value == "" {
		t.Skip("set GO_PHERENCE_POCKETTTS_RELEASED_TRAINING_STEPS")
	}
	steps, err := strconv.Atoi(value)
	if err != nil || steps <= 0 || steps > 1000 {
		t.Fatalf("invalid released training steps %q", value)
	}
	textTokens := 32
	if text := os.Getenv("GO_PHERENCE_POCKETTTS_RELEASED_TRAINING_TEXT_TOKENS"); text != "" {
		textTokens, err = strconv.Atoi(text)
		if err != nil || textTokens < 1 || textTokens > 512 {
			t.Fatalf("invalid released training text tokens %q", text)
		}
	}
	lm, flow, weighting, batch, samples, plan := releasedTrainingRow(t, textTokens)
	if textTokens == 512 && plan.SequenceRows != 950 {
		t.Fatalf("released maximum-text sequence rows=%d want=950", plan.SequenceRows)
	}
	t.Logf("released training text_tokens=%d sequence_rows=%d", textTokens, plan.SequenceRows)
	workspace, err := NewAdmittedTrainingWorkspace(lm, flow, weighting, batch, plan)
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := NewFullTrainer(lm, flow, weighting, DefaultAdamWConfig(), .999)
	if err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	for step := 0; step < steps; step++ {
		for row := 0; row < batch.Frames; row++ {
			samples.DiagonalTime[row] = .1 + .8*float32((row+step)%29)/28
			samples.DistillS[row] = .05 + .35*float32((row+3*step)%23)/22
			samples.DistillT[row] = samples.DistillS[row] + .55
			for channel := 0; channel < lm.LatentDim; channel++ {
				i := row*lm.LatentDim + channel
				samples.Noise[i] = float32(((i+step*17)%37)-18) / 37
			}
		}
		metrics, gradients, err := PocketTrainingStepInto(lm, flow, weighting, batch, samples, DefaultTrainingStepConfig(), workspace)
		if err != nil {
			t.Fatalf("step %d forward/backward: %v", step+1, err)
		}
		if !isFinite(float32(metrics.Loss)) || !isFinite(float32(metrics.FlowLoss)) || !isFinite(float32(metrics.EOS)) {
			t.Fatalf("step %d non-finite metrics %+v", step+1, metrics)
		}
		if workspace.primaryFlowTape.fallbacks != 0 || workspace.endpointFlowTape.fallbacks != 0 || workspace.primaryFlowBackward.fallbacks != 0 || workspace.endpointFlowBackward.fallbacks != 0 {
			t.Fatalf("step %d flow scratch spilled: primary=%d endpoint=%d backward=%d endpoint_backward=%d", step+1, workspace.primaryFlowTape.fallbacks, workspace.endpointFlowTape.fallbacks, workspace.primaryFlowBackward.fallbacks, workspace.endpointFlowBackward.fallbacks)
		}
		if err = trainer.Step(gradients); err != nil {
			t.Fatalf("step %d optimiser: %v", step+1, err)
		}
		if (step+1)%10 == 0 || step+1 == steps {
			runtime.GC()
			var memory runtime.MemStats
			runtime.ReadMemStats(&memory)
			t.Logf("released production training step %d/%d elapsed=%s loss=%.7g heap_alloc=%d heap_inuse=%d objects=%d", step+1, steps, time.Since(started).Round(time.Millisecond), metrics.Loss, memory.HeapAlloc, memory.HeapInuse, memory.HeapObjects)
		}
	}
	params, err := fullParameterMap(lm, flow, weighting)
	if err != nil {
		t.Fatal(err)
	}
	for name, values := range params {
		if !finiteF32(values) {
			t.Fatalf("non-finite parameter %q after %d steps", name, steps)
		}
	}
	for name, state := range trainer.Moments {
		if !finiteF32(state.M) || !finiteF32(state.V) {
			t.Fatalf("non-finite optimiser state %q after %d steps", name, steps)
		}
	}
	for name, values := range trainer.EMA {
		if !finiteF32(values) {
			t.Fatalf("non-finite EMA %q after %d steps", name, steps)
		}
	}
	runtime.GC()
	var memory runtime.MemStats
	runtime.ReadMemStats(&memory)
	t.Logf("released production training complete steps=%d elapsed=%s heap_alloc=%d heap_inuse=%d objects=%d", steps, time.Since(started).Round(time.Millisecond), memory.HeapAlloc, memory.HeapInuse, memory.HeapObjects)
}
