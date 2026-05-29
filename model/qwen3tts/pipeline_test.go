package qwen3tts

import "testing"

func TestPipelinePlanByVariant(t *testing.T) {
	for _, tc := range []struct {
		modelType ModelType
		want      ConditioningMode
	}{
		{CustomVoice, ConditioningCustomVoice},
		{Base, ConditioningReferenceAudio},
		{VoiceDesign, ConditioningVoiceDesign},
	} {
		plan, err := NewPipelinePlan(ParsedConfig{ModelType: tc.modelType})
		if err != nil {
			t.Fatal(err)
		}
		if plan.Conditioning != tc.want {
			t.Fatalf("%s conditioning=%s want=%s", tc.modelType, plan.Conditioning, tc.want)
		}
		if len(plan.Steps) != 5 || plan.Steps[0].Stage != StageConditioning || plan.Steps[4].Stage != StageDecoder12Hz {
			t.Fatalf("plan=%+v", plan)
		}
	}
}

func TestPipelinePlanRejectsMalformed(t *testing.T) {
	p := PipelinePlan{Steps: []PipelineStep{{Index: 1, Stage: StageConditioning}}}
	if err := p.Validate(); err == nil {
		t.Fatal("expected length/index error")
	}
	p = PipelinePlan{Steps: []PipelineStep{
		{Index: 0, Stage: StageConditioning},
		{Index: 1, Stage: StageTalker},
		{Index: 2, Stage: StagePromptPrefill},
		{Index: 3, Stage: StageCodePredictor},
		{Index: 4, Stage: StageDecoder12Hz},
	}}
	if err := p.Validate(); err == nil {
		t.Fatal("expected stage order error")
	}
}
