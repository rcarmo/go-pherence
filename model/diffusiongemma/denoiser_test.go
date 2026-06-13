package diffusiongemma

import "testing"

type captureDispatcher struct {
	ctx ForwardContext
}

func (d *captureDispatcher) RunTextForward(ctx ForwardContext, _ *TextWeights, _ ForwardOpPlan, _ ForwardBufferPlan) (ForwardOutput, error) {
	d.ctx = ctx
	return ForwardOutput{Logits: [][]float32{{0}}}, nil
}

func TestTextDenoiserClearsStaleEncoderKVForUnsupportedDispatcher(t *testing.T) {
	disp := &captureDispatcher{}
	d := &TextDenoiser{
		Shape:            Shape{VocabSize: 10},
		Dispatcher:       disp,
		EncoderKV:        []EncoderKVLayer{{SeqLen: 2, KVHeads: 1, HeadDim: 1, Keys: []float32{1, 2}, Values: []float32{3, 4}}},
		EncoderPromptIDs: []int{1, 2},
	}
	_, err := d.Denoise(ForwardInput{PromptIDs: []int{1, 2, 3}, Canvas: []int{9}})
	if err != nil {
		t.Fatal(err)
	}
	if d.EncoderKV != nil || d.EncoderPromptIDs != nil {
		t.Fatalf("expected stale encoder cache to be cleared, got kv=%v prompt=%v", d.EncoderKV, d.EncoderPromptIDs)
	}
	if len(disp.ctx.EncoderKV) != 0 || disp.ctx.EncoderSeqLen != 0 {
		t.Fatalf("dispatcher received stale encoder context: kv=%d seq=%d", len(disp.ctx.EncoderKV), disp.ctx.EncoderSeqLen)
	}
}

func TestTextDenoiserKeepsEncoderKVOnPromptCacheHit(t *testing.T) {
	disp := &captureDispatcher{}
	kv := []EncoderKVLayer{{SeqLen: 2, KVHeads: 1, HeadDim: 1, Keys: []float32{1, 2}, Values: []float32{3, 4}}}
	d := &TextDenoiser{
		Shape:            Shape{VocabSize: 10},
		Dispatcher:       disp,
		EncoderKV:        kv,
		EncoderPromptIDs: []int{1, 2},
	}
	_, err := d.Denoise(ForwardInput{PromptIDs: []int{1, 2}, Canvas: []int{9}})
	if err != nil {
		t.Fatal(err)
	}
	if len(d.EncoderKV) != 1 || len(d.EncoderPromptIDs) != 2 {
		t.Fatalf("expected encoder cache to remain, got kv=%v prompt=%v", d.EncoderKV, d.EncoderPromptIDs)
	}
	if len(disp.ctx.EncoderKV) != 1 || disp.ctx.EncoderSeqLen != 2 {
		t.Fatalf("dispatcher did not receive cached encoder context: kv=%d seq=%d", len(disp.ctx.EncoderKV), disp.ctx.EncoderSeqLen)
	}
}
