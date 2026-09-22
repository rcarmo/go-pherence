package pockettts

import (
	"math"
	"testing"
)

func tinyFlowLMTraining() *FlowLMTrainingCPU {
	transformer := tinyTrainableTransformer()
	transformer.Context = 0
	return &FlowLMTrainingCPU{
		Vocabulary: 6, Hidden: 4, LatentDim: 2,
		Embedding: func() []float32 {
			values := make([]float32, 24)
			for i := range values {
				values[i] = float32((i*5)%17-8) * .03
			}
			return values
		}(),
		BOS: []float32{.15, -.25}, BOSBeforeVoice: []float32{.05, -.1, .2, -.3},
		SpeakerProjection: func() LinearF32 { l := tinyLinear(4, 2, .02); l.Bias = nil; return l }(),
		Input:             func() LinearF32 { l := tinyLinear(4, 2, -.015); l.Bias = nil; return l }(),
		Transformer:       transformer,
		EOS:               tinyLinear(1, 4, .01),
	}
}

func tinyFlowLMBatch() FlowLMTrainingBatch {
	return FlowLMTrainingBatch{Frames: 3, VoiceFrames: 2, NormalizedLatents: []float32{.2, -.4, .6, .1, -.3, .7}, VoiceLatents: []float32{-.2, .5, .8, -.1}, TextTokens: []uint32{1, 3}}
}

func TestFlowLMTrainingFiniteDifference(t *testing.T) {
	model, batch := tinyFlowLMTraining(), tinyFlowLMBatch()
	dZ := []float32{.1, -.2, .3, -.4, .5, -.6, .7, -.8, -.3, .4, -.5, .6}
	dEOS := []float32{.2, -.35, .45}
	output, gradients, inputs, err := model.ForwardBackward(batch, dZ, dEOS)
	if err != nil {
		t.Fatal(err)
	}
	if output.PrefixRows != 5 || output.SequenceRows != 8 || len(output.Z) != 12 || len(output.EOS) != 3 {
		t.Fatalf("unexpected output: %+v", output)
	}
	evaluate := func() float64 {
		value, _, _, e := model.ForwardBackward(batch, make([]float32, len(dZ)), make([]float32, len(dEOS)))
		if e != nil {
			t.Fatal(e)
		}
		total := float64(0)
		for i := range value.Z {
			total += float64(value.Z[i]) * float64(dZ[i])
		}
		for i := range value.EOS {
			total += float64(value.EOS[i]) * float64(dEOS[i])
		}
		return total
	}
	checkCentralDifferenceWithStep(t, "flowlm.audio", batch.NormalizedLatents, inputs.NormalizedLatents, evaluate, 2e-4, 1.5e-2)
	checkCentralDifferenceWithStep(t, "flowlm.voice", batch.VoiceLatents, inputs.VoiceLatents, evaluate, 2e-4, 1.5e-2)
	params, grads := flowLMTrainingParameterPairs(model, gradients)
	for i := range params {
		checkCentralDifferenceWithStep(t, params[i].name, params[i].values, grads[i].values, evaluate, 2e-4, 2e-2)
	}
	// The final target latent is never teacher-forced into the shifted audio input.
	if inputs.NormalizedLatents[4] != 0 || inputs.NormalizedLatents[5] != 0 {
		t.Fatalf("last target leaked into input gradient: %v", inputs.NormalizedLatents)
	}
}

func TestFlowLMTrainingRepeatedTokenGradientAccumulates(t *testing.T) {
	model, batch := tinyFlowLMTraining(), tinyFlowLMBatch()
	batch.TextTokens = []uint32{2, 2}
	dZ := make([]float32, batch.Frames*model.Hidden)
	dZ[0] = 1
	_, gradients, _, err := model.ForwardBackward(batch, dZ, make([]float32, batch.Frames))
	if err != nil {
		t.Fatal(err)
	}
	nonzero := false
	for _, value := range gradients.Embedding[2*model.Hidden : 3*model.Hidden] {
		if value != 0 {
			nonzero = true
		}
	}
	if !nonzero {
		t.Fatal("repeated token row gradient is zero")
	}
	for token := 0; token < model.Vocabulary; token++ {
		if token == 2 {
			continue
		}
		for _, value := range gradients.Embedding[token*model.Hidden : (token+1)*model.Hidden] {
			if value != 0 {
				t.Fatalf("unused token %d received gradient", token)
			}
		}
	}
}

func TestFlowLMTrainingRejectsMalformed(t *testing.T) {
	model, batch := tinyFlowLMTraining(), tinyFlowLMBatch()
	if _, _, _, err := model.ForwardBackward(batch, nil, nil); err == nil {
		t.Fatal("accepted missing seeds")
	}
	batch.TextTokens[0] = uint32(model.Vocabulary)
	if _, _, _, err := model.ForwardBackward(batch, make([]float32, 12), make([]float32, 3)); err == nil {
		t.Fatal("accepted bad token")
	}
	batch.TextTokens[0] = ^uint32(0)
	if _, _, _, err := model.ForwardBackward(batch, make([]float32, 12), make([]float32, 3)); err == nil {
		t.Fatal("accepted narrowing-overflow token")
	}
	batch = tinyFlowLMBatch()
	batch.NormalizedLatents[0] = float32(math.NaN())
	if _, _, _, err := model.ForwardBackward(batch, make([]float32, 12), make([]float32, 3)); err == nil {
		t.Fatal("accepted non-finite audio")
	}
	model = tinyFlowLMTraining()
	model.Input.WeightBF16 = make([]uint16, len(model.Input.Weight))
	if _, _, _, err := model.ForwardBackward(tinyFlowLMBatch(), make([]float32, 12), make([]float32, 3)); err == nil {
		t.Fatal("accepted BF16 input projection")
	}
}

func checkCentralDifferenceWithStep(t *testing.T, name string, values, gradients []float32, evaluate func() float64, step float32, tolerance float64) {
	t.Helper()
	if len(values) != len(gradients) {
		t.Fatalf("%s shape=%d gradient=%d", name, len(values), len(gradients))
	}
	for i := range values {
		original := values[i]
		values[i] = original + step
		plus := evaluate()
		values[i] = original - step
		minus := evaluate()
		values[i] = original
		numeric := (plus - minus) / (2 * float64(step))
		if math.Abs(float64(gradients[i])-numeric) > tolerance {
			t.Fatalf("%s[%d] gradient=%g numeric=%g diff=%g", name, i, gradients[i], numeric, math.Abs(float64(gradients[i])-numeric))
		}
	}
}

func flowLMTrainingParameterPairs(model *FlowLMTrainingCPU, gradients *FlowLMTrainingGradients) ([]parameterPair, []parameterPair) {
	params := []parameterPair{{"flowlm.embedding", model.Embedding}, {"flowlm.bos", model.BOS}, {"flowlm.bos_before_voice", model.BOSBeforeVoice}, {"flowlm.speaker.weight", model.SpeakerProjection.Weight}, {"flowlm.input.weight", model.Input.Weight}, {"flowlm.eos.weight", model.EOS.Weight}, {"flowlm.eos.bias", model.EOS.Bias}}
	grads := []parameterPair{{"flowlm.embedding", gradients.Embedding}, {"flowlm.bos", gradients.BOS}, {"flowlm.bos_before_voice", gradients.BOSBeforeVoice}, {"flowlm.speaker.weight", gradients.SpeakerProjection.Weight}, {"flowlm.input.weight", gradients.Input.Weight}, {"flowlm.eos.weight", gradients.EOS.Weight}, {"flowlm.eos.bias", gradients.EOS.Bias}}
	tp, tg := transformerParameterPairs(model.Transformer, gradients.Transformer)
	params = append(params, tp...)
	grads = append(grads, tg...)
	return params, grads
}
