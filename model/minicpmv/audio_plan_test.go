package minicpmv

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/config"
)

func TestClassifyAudioTensorName(t *testing.T) {
	cases := map[string]AudioTensorRole{
		"audio_encoder.conv1.weight":                     AudioConv,
		"audio_encoder.layers.0.self_attn.q_proj.weight": AudioAttention,
		"audio_encoder.layers.0.mlp.fc1.weight":          AudioMLP,
		"audio_encoder.layers.0.layer_norm.weight":       AudioNorm,
		"audio_projector.weight":                         AudioProjector,
	}
	for name, want := range cases {
		if got := ClassifyAudioTensorName(name); got != want {
			t.Fatalf("ClassifyAudioTensorName(%q)=%s want %s", name, got, want)
		}
	}
}

func TestBuildAudioExecutionPlan(t *testing.T) {
	summary := config.MiniCPMVSummary{AudioModelType: "whisper_encoder", AudioHiddenSize: 1280, AudioLayers: 32, AudioHeads: 20, AudioFeatureSize: 128, AudioMelBins: 128, AudioSamplingRate: 16000}
	plan := BuildAudioExecutionPlan(summary, []string{
		"audio_encoder.conv1.weight",
		"audio_encoder.layers.0.self_attn.q_proj.weight",
		"audio_encoder.layers.0.mlp.fc1.weight",
	})
	if !plan.MetadataReady || !plan.TensorReady || plan.Ready || plan.Counts[AudioConv] != 1 || plan.Counts[AudioAttention] != 1 {
		t.Fatalf("bad audio plan: %+v", plan)
	}
	if got := findAudioOp(plan, "audio_encoder_execution"); got == nil || got.Ready || got.Reason == "" {
		t.Fatalf("audio execution should be pending: %+v", plan.Ops)
	}
}

func TestBuildAudioExecutionPlanMissingMetadata(t *testing.T) {
	plan := BuildAudioExecutionPlan(config.MiniCPMVSummary{}, nil)
	if plan.MetadataReady || plan.TensorReady || plan.Ready {
		t.Fatalf("unexpected ready audio plan: %+v", plan)
	}
	if got := findAudioOp(plan, "audio_metadata"); got == nil || got.Ready || got.Reason == "" {
		t.Fatalf("missing metadata op reason: %+v", plan.Ops)
	}
}

func findAudioOp(plan AudioExecutionPlan, name string) *AudioOp {
	for i := range plan.Ops {
		if plan.Ops[i].Name == name {
			return &plan.Ops[i]
		}
	}
	return nil
}
