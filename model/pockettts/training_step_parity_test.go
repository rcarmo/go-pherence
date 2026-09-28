package pockettts

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
)

type trainingStepOracle struct {
	Schema               int                  `json:"schema"`
	UpstreamRevision     string               `json:"upstream_revision"`
	Seed                 int                  `json:"seed"`
	Frames               int                  `json:"frames"`
	VoiceFrames          int                  `json:"voice_frames"`
	Hidden               int                  `json:"hidden"`
	LatentDim            int                  `json:"latent_dim"`
	Vocabulary           int                  `json:"vocabulary"`
	Metrics              map[string]float64   `json:"metrics"`
	NormalizedLatents    []float32            `json:"normalized_latents"`
	VoiceLatents         []float32            `json:"voice_latents"`
	TextTokens           []uint32             `json:"text_tokens"`
	Mask                 []bool               `json:"mask"`
	Noise                []float32            `json:"noise"`
	DiagonalTime         []float32            `json:"diagonal_time"`
	DistillS             []float32            `json:"distill_s"`
	DistillT             []float32            `json:"distill_t"`
	DiagonalLogVariance  []float32            `json:"diagonal_log_variance"`
	DistillLogVariance   []float32            `json:"distill_log_variance"`
	DDiagonalLogVariance []float32            `json:"d_diagonal_log_variance"`
	DDistillLogVariance  []float32            `json:"d_distill_log_variance"`
	DNormalizedLatents   []float32            `json:"d_normalized_latents"`
	DVoiceLatents        []float32            `json:"d_voice_latents"`
	Parameters           map[string][]float32 `json:"parameters"`
	Gradients            map[string][]float32 `json:"gradients"`
	AdamW                struct {
		LearningRate        float32              `json:"learning_rate"`
		Beta1               float32              `json:"beta1"`
		Beta2               float32              `json:"beta2"`
		Epsilon             float32              `json:"epsilon"`
		WeightDecay         float32              `json:"weight_decay"`
		ParametersAfterStep map[string][]float32 `json:"parameters_after_step"`
		EMADecay            float32              `json:"ema_decay"`
		EMAAfterStep        map[string][]float32 `json:"ema_after_step"`
	} `json:"adamw"`
}

func TestPocketTrainingStepPyTorchParity(t *testing.T) {
	data, err := os.ReadFile("testdata/training_step_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle trainingStepOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Schema != 1 || oracle.UpstreamRevision != UpstreamCommit || oracle.Frames <= 0 || oracle.Hidden <= 0 {
		t.Fatal("unexpected step oracle")
	}
	flowLM, flow, weighting := trainingStepModelsFromOracle(t, oracle)
	batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: oracle.VoiceFrames, NormalizedLatents: oracle.NormalizedLatents, VoiceLatents: oracle.VoiceLatents, TextTokens: oracle.TextTokens}
	samples := TrainingStepSamples{Mask: oracle.Mask, Noise: oracle.Noise, DiagonalTime: oracle.DiagonalTime, DistillS: oracle.DistillS, DistillT: oracle.DistillT}
	metrics, gradients, err := PocketTrainingStep(flowLM, flow, weighting, batch, samples, DefaultTrainingStepConfig())
	if err != nil {
		t.Fatal(err)
	}
	assertClose64(t, "step diagonal", metrics.FlowDiagonal, oracle.Metrics["flow_diag"], 3e-5)
	assertClose64(t, "step distill", metrics.FlowDistill, oracle.Metrics["flow_distill"], 3e-5)
	assertClose64(t, "step eos", metrics.EOS, oracle.Metrics["eos_loss"], 3e-5)
	assertClose64(t, "step flow", metrics.FlowLoss, oracle.Metrics["flow_loss"], 3e-5)
	assertClose64(t, "step loss", metrics.Loss, oracle.Metrics["loss"], 3e-5)
	assertSliceClose(t, "step diagonal logvars", gradients.DiagonalLogVariance[:2], oracle.DDiagonalLogVariance, 5e-5)
	assertSliceClose(t, "step distill logvars", gradients.DistillLogVariance[:2], oracle.DDistillLogVariance, 5e-5)
	if gradients.DiagonalLogVariance[2] != 0 || gradients.DistillLogVariance[2] != 0 {
		t.Fatal("masked log-variance row received gradient")
	}
	assertSliceClose(t, "step audio", gradients.Inputs.NormalizedLatents, oracle.DNormalizedLatents, 5e-5)
	assertSliceClose(t, "step voice", gradients.Inputs.VoiceLatents, oracle.DVoiceLatents, 5e-5)
	got := trainingStepGradientMap(gradients)
	if len(got) != len(oracle.Gradients) {
		t.Fatalf("step gradient count=%d want=%d", len(got), len(oracle.Gradients))
	}
	for name, values := range got {
		want, ok := oracle.Gradients[name]
		if !ok {
			t.Fatalf("oracle lacks %s", name)
		}
		assertSliceClose(t, "step "+name, values, want, 8e-5)
	}
}

func trainingStepModelsFromOracle(t testing.TB, oracle trainingStepOracle) (*FlowLMTrainingCPU, *FlowHeadCPU, *LSDWeightMLP) {
	t.Helper()
	lm, fh := map[string][]float32{}, map[string][]float32{}
	for name, values := range oracle.Parameters {
		if strings.HasPrefix(name, "flow_lm.flow_net.") {
			fh[strings.TrimPrefix(name, "flow_lm.flow_net.")] = values
		} else if strings.HasPrefix(name, "flow_lm.") {
			lm[strings.TrimPrefix(name, "flow_lm.")] = values
		}
	}
	flowLM := flowLMTrainingFromOracle(t, flowLMTrainingOracle{Hidden: oracle.Hidden, LatentDim: oracle.LatentDim, Vocabulary: oracle.Vocabulary, Parameters: lm})
	flow := flowHeadFromOracleDims(t, fh, oracle.Hidden)
	weighting := &LSDWeightMLP{Layers: []LinearF32{
		{Weight: append([]float32(nil), oracle.Parameters["flow.w_s_t.0.weight"]...), Bias: append([]float32(nil), oracle.Parameters["flow.w_s_t.0.bias"]...), In: 2, Out: 3},
		{Weight: append([]float32(nil), oracle.Parameters["flow.w_s_t.2.weight"]...), Bias: append([]float32(nil), oracle.Parameters["flow.w_s_t.2.bias"]...), In: 3, Out: 3},
		{Weight: append([]float32(nil), oracle.Parameters["flow.w_s_t.4.weight"]...), Bias: append([]float32(nil), oracle.Parameters["flow.w_s_t.4.bias"]...), In: 3, Out: 1},
	}}
	params, err := fullParameterMap(flowLM, flow, weighting)
	if err != nil {
		t.Fatal(err)
	}
	buffers := fullBufferMap(flowLM, flow)
	if len(params)+len(buffers) != len(oracle.Parameters) {
		t.Fatalf("native state entries=%d want=%d", len(params)+len(buffers), len(oracle.Parameters))
	}
	for name, want := range oracle.Parameters {
		got, ok := params[name]
		if !ok {
			got, ok = buffers[name]
		}
		if !ok {
			t.Fatalf("unused oracle state entry %s", name)
		}
		assertSliceClose(t, "oracle state "+name, got, want, 0)
	}
	return flowLM, flow, weighting
}

func trainingStepGradientMap(g TrainingStepGradients) map[string][]float32 {
	out := map[string][]float32{}
	for name, values := range flowLMTrainingGradientMap(g.FlowLM) {
		out["flow_lm."+name] = values
	}
	for name, values := range flowHeadGradientMap(g.Flow) {
		out["flow_lm.flow_net."+name] = values
	}
	indices := []int{0, 2, 4}
	for i, layer := range g.Weighting.Layers {
		prefix := fmt.Sprintf("flow.w_s_t.%d", indices[i])
		out[prefix+".weight"], out[prefix+".bias"] = layer.Weight, layer.Bias
	}
	return out
}
