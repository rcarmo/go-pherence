package qwen3tts

import "fmt"

type PipelineStage string

const (
	StageConditioning  PipelineStage = "conditioning"
	StagePromptPrefill PipelineStage = "prompt_prefill"
	StageTalker        PipelineStage = "talker"
	StageCodePredictor PipelineStage = "code_predictor"
	StageDecoder12Hz   PipelineStage = "decoder12hz"
)

type PipelineStep struct {
	Index       int           `json:"index"`
	Stage       PipelineStage `json:"stage"`
	Description string        `json:"description"`
}

type PipelinePlan struct {
	Conditioning ConditioningMode `json:"conditioning"`
	Steps        []PipelineStep   `json:"steps"`
}

func NewPipelinePlan(cfg ParsedConfig) (PipelinePlan, error) {
	caps, err := cfg.Capabilities()
	if err != nil {
		return PipelinePlan{}, err
	}
	steps := []PipelineStep{{Index: 0, Stage: StageConditioning, Description: string(caps.Conditioning)}}
	switch caps.Conditioning {
	case ConditioningCustomVoice:
		steps[0].Description = "fixed speaker/language control tokens"
	case ConditioningReferenceAudio:
		steps[0].Description = "reference audio codec/speaker conditioning"
	case ConditioningVoiceDesign:
		steps[0].Description = "text voice-design prompt conditioning"
	default:
		return PipelinePlan{}, fmt.Errorf("unknown Qwen3-TTS conditioning mode %q", caps.Conditioning)
	}
	steps = append(steps,
		PipelineStep{Index: 1, Stage: StagePromptPrefill, Description: "tokenized text/control streams to talker embeddings"},
		PipelineStep{Index: 2, Stage: StageTalker, Description: "semantic token generation"},
		PipelineStep{Index: 3, Stage: StageCodePredictor, Description: "15-code acoustic frame prediction"},
		PipelineStep{Index: 4, Stage: StageDecoder12Hz, Description: "12Hz codec frames to waveform"},
	)
	plan := PipelinePlan{Conditioning: caps.Conditioning, Steps: steps}
	return plan, plan.Validate()
}

func (p PipelinePlan) Validate() error {
	if len(p.Steps) != 5 {
		return fmt.Errorf("invalid Qwen3-TTS pipeline length=%d", len(p.Steps))
	}
	want := []PipelineStage{StageConditioning, StagePromptPrefill, StageTalker, StageCodePredictor, StageDecoder12Hz}
	for i, step := range p.Steps {
		if step.Index != i {
			return fmt.Errorf("invalid Qwen3-TTS pipeline index at position=%d index=%d", i, step.Index)
		}
		if step.Stage != want[i] {
			return fmt.Errorf("invalid Qwen3-TTS pipeline stage at %d: got=%s want=%s", i, step.Stage, want[i])
		}
	}
	return nil
}
