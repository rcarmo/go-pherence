package pockettts

import (
	"encoding/json"
	"os"
	"testing"
)

type flowLMTrainingOracle struct {
	Schema             int                  `json:"schema"`
	UpstreamRevision   string               `json:"upstream_revision"`
	Frames             int                  `json:"frames"`
	VoiceFrames        int                  `json:"voice_frames"`
	Hidden             int                  `json:"hidden"`
	LatentDim          int                  `json:"latent_dim"`
	Vocabulary         int                  `json:"vocabulary"`
	NormalizedLatents  []float32            `json:"normalized_latents"`
	VoiceLatents       []float32            `json:"voice_latents"`
	TextTokens         []uint32             `json:"text_tokens"`
	DZ                 []float32            `json:"d_z"`
	DEOS               []float32            `json:"d_eos"`
	PrefixRows         int                  `json:"prefix_rows"`
	Sequence           []float32            `json:"sequence"`
	Z                  []float32            `json:"z"`
	EOS                []float32            `json:"eos"`
	DNormalizedLatents []float32            `json:"d_normalized_latents"`
	DVoiceLatents      []float32            `json:"d_voice_latents"`
	Parameters         map[string][]float32 `json:"parameters"`
	Gradients          map[string][]float32 `json:"gradients"`
}

func TestFlowLMTrainingPyTorchParity(t *testing.T) {
	data, err := os.ReadFile("testdata/flowlm_training_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle flowLMTrainingOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Schema != 1 || oracle.UpstreamRevision != UpstreamCommit {
		t.Fatal("unexpected FlowLM training oracle")
	}
	model := flowLMTrainingFromOracle(t, oracle)
	batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: oracle.VoiceFrames, NormalizedLatents: oracle.NormalizedLatents, VoiceLatents: oracle.VoiceLatents, TextTokens: oracle.TextTokens}
	output, gradients, inputs, err := model.ForwardBackward(batch, oracle.DZ, oracle.DEOS)
	if err != nil {
		t.Fatal(err)
	}
	if output.PrefixRows != oracle.PrefixRows {
		t.Fatalf("prefix=%d want=%d", output.PrefixRows, oracle.PrefixRows)
	}
	assertSliceClose(t, "FlowLM PyTorch sequence", output.Sequence, oracle.Sequence, 2e-6)
	assertSliceClose(t, "FlowLM PyTorch z", output.Z, oracle.Z, 5e-6)
	assertSliceClose(t, "FlowLM PyTorch eos", output.EOS, oracle.EOS, 5e-6)
	assertSliceClose(t, "FlowLM PyTorch audio", inputs.NormalizedLatents, oracle.DNormalizedLatents, 1.5e-5)
	assertSliceClose(t, "FlowLM PyTorch voice", inputs.VoiceLatents, oracle.DVoiceLatents, 1.5e-5)
	gradientMap := flowLMTrainingGradientMap(gradients)
	if len(gradientMap) != len(oracle.Gradients) {
		t.Fatalf("gradient count=%d want=%d", len(gradientMap), len(oracle.Gradients))
	}
	for name, got := range gradientMap {
		want, ok := oracle.Gradients[name]
		if !ok {
			t.Fatalf("oracle lacks %s", name)
		}
		assertSliceClose(t, "FlowLM PyTorch "+name, got, want, 2e-5)
	}
}

func flowLMTrainingFromOracle(t *testing.T, oracle flowLMTrainingOracle) *FlowLMTrainingCPU {
	t.Helper()
	p := oracle.Parameters
	linear := func(prefix string, out, in int, bias bool) LinearF32 {
		w := p[prefix+".weight"]
		if len(w) != out*in {
			t.Fatalf("%s weight=%d", prefix, len(w))
		}
		l := LinearF32{Weight: append([]float32(nil), w...), In: in, Out: out}
		if bias {
			l.Bias = append([]float32(nil), p[prefix+".bias"]...)
		}
		return l
	}
	transformerOracle := transformerBackwardOracle{Width: oracle.Hidden, Heads: 2, Context: 0, MaxPeriod: 100, Parameters: map[string][]float32{}}
	for name, value := range p {
		if len(name) > 12 && name[:12] == "transformer." {
			transformerOracle.Parameters[name] = value
		}
	}
	transformerOracle.Parameters["final.weight"], transformerOracle.Parameters["final.bias"] = p["out_norm.weight"], p["out_norm.bias"]
	transformer := transformerFromOracle(t, transformerOracle)
	return &FlowLMTrainingCPU{Embedding: append([]float32(nil), p["conditioner.embed.weight"]...), Vocabulary: oracle.Vocabulary, Hidden: oracle.Hidden, LatentDim: oracle.LatentDim, BOS: append([]float32(nil), p["bos_emb"]...), BOSBeforeVoice: append([]float32(nil), p["bos_before_voice"]...), LatentMean: append([]float32(nil), p["emb_mean"]...), LatentStd: append([]float32(nil), p["emb_std"]...), SpeakerProjection: LinearF32{Weight: append([]float32(nil), p["speaker_proj_weight"]...), In: oracle.LatentDim, Out: oracle.Hidden}, Input: linear("input_linear", oracle.Hidden, oracle.LatentDim, false), Transformer: transformer, EOS: linear("out_eos", 1, oracle.Hidden, true)}
}

func flowLMTrainingGradientMap(g *FlowLMTrainingGradients) map[string][]float32 {
	out := map[string][]float32{"conditioner.embed.weight": g.Embedding, "bos_emb": g.BOS, "bos_before_voice": g.BOSBeforeVoice, "speaker_proj_weight": g.SpeakerProjection.Weight, "input_linear.weight": g.Input.Weight, "out_eos.weight": g.EOS.Weight, "out_eos.bias": g.EOS.Bias, "out_norm.weight": g.Transformer.FinalWeight, "out_norm.bias": g.Transformer.FinalBias}
	for name, value := range transformerGradientMap(g.Transformer) {
		if name == "final.weight" || name == "final.bias" {
			continue
		}
		out[name] = value
	}
	return out
}
