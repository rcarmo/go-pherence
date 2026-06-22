package minicpmv

import "testing"

func TestClassifyTensorName(t *testing.T) {
	cases := map[string]TensorGroup{
		"llm.model.embed_tokens.weight":                             TensorTextEmbedding,
		"llm.model.layers.0.self_attn.q_proj.weight":                TensorTextLayer,
		"llm.lm_head.weight":                                        TensorTextLMHead,
		"vpm.encoder.layers.0.self_attn.q_proj.weight":              TensorVisionTower,
		"vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight": TensorVisionTower,
		"audio_encoder.layers.0.self_attn.q_proj.weight":            TensorAudioEncoder,
		"resampler.query.weight":                                    TensorResampler,
		"resampler.pos_embed":                                       TensorResampler,
		"mm_projector.0.weight":                                     TensorProjector,
		"model.norm.weight":                                         TensorNorm,
	}
	for name, want := range cases {
		if got := ClassifyTensorName(name); got != want {
			t.Fatalf("ClassifyTensorName(%q)=%s want %s", name, got, want)
		}
	}
}

func TestSummarizeTensorsAndReadiness(t *testing.T) {
	inv := SummarizeTensors([]string{
		"llm.model.embed_tokens.weight",
		"llm.model.layers.0.self_attn.q_proj.weight",
		"llm.model.layers.0.mlp.up_proj.weight",
		"llm.lm_head.weight",
		"vpm.encoder.layers.0.self_attn.q_proj.weight",
		"resampler.query.weight",
	})
	if inv.Total != 6 || inv.Groups[TensorTextLayer] != 2 || inv.Groups[TensorVisionTower] != 1 || inv.Groups[TensorResampler] != 1 {
		t.Fatalf("bad inventory: %+v", inv)
	}
	ready := TensorReadinessFromInventory(inv)
	if !ready.HasTextEmbedding || !ready.HasTextLayers || !ready.HasVisionTower || !ready.HasResampler || !ready.MetadataReady || ready.RuntimeReady {
		t.Fatalf("bad readiness: %+v", ready)
	}
}
