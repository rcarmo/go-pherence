package qwen3tts

import (
	"encoding/json"
	"fmt"
	"os"
)

// ReferenceFixture is a small, commit-safe parity anchor for Qwen3-TTS. Fields
// beyond Prompt are optional until qwen3-tts-rs/Transformers traces are captured;
// keeping them in the schema now prevents ad-hoc fixture formats later.
type ReferenceFixture struct {
	Name          string                  `json:"name"`
	Variant       ModelType               `json:"variant"`
	ModelSize     string                  `json:"model_size"`
	Text          string                  `json:"text"`
	Speaker       Speaker                 `json:"speaker"`
	Language      Language                `json:"language"`
	Prompt        PromptIDs               `json:"prompt"`
	Talker        *TalkerReference        `json:"talker,omitempty"`
	CodePredictor *CodePredictorReference `json:"code_predictor,omitempty"`
	Decoder12Hz   *Decoder12HzReference   `json:"decoder12hz,omitempty"`
}

type TalkerReference struct {
	FirstSemanticToken uint32  `json:"first_semantic_token"`
	LogitChecksum      string  `json:"logit_checksum,omitempty"`
	HiddenChecksum     string  `json:"hidden_checksum,omitempty"`
	MaxAbsDiff         float64 `json:"max_abs_diff,omitempty"`
}

type CodePredictorReference struct {
	AcousticFrame []uint32 `json:"acoustic_frame"`
	LogitChecksum string   `json:"logit_checksum,omitempty"`
	MaxAbsDiff    float64  `json:"max_abs_diff,omitempty"`
}

type Decoder12HzReference struct {
	SampleRate int     `json:"sample_rate"`
	Samples    int     `json:"samples"`
	DurationS  float64 `json:"duration_s"`
	SHA256     string  `json:"sha256,omitempty"`
	RMS        float64 `json:"rms,omitempty"`
}

type ReferenceCoverage struct {
	Prompt               bool     `json:"prompt"`
	SemanticToken        bool     `json:"semantic_token"`
	AcousticFrame        bool     `json:"acoustic_frame"`
	DecodedWAVSummary    bool     `json:"decoded_wav_summary"`
	CompleteRuntimeTrace bool     `json:"complete_runtime_trace"`
	Missing              []string `json:"missing,omitempty"`
}

func LoadReferenceFixture(path string) (ReferenceFixture, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ReferenceFixture{}, err
	}
	var fx ReferenceFixture
	if err := json.Unmarshal(data, &fx); err != nil {
		return ReferenceFixture{}, err
	}
	return fx, fx.Validate()
}

func (fx ReferenceFixture) Coverage() ReferenceCoverage {
	cov := ReferenceCoverage{Prompt: len(fx.Prompt.Text) > 0 && len(fx.Prompt.Codec) > 0}
	cov.SemanticToken = fx.Talker != nil
	cov.AcousticFrame = fx.CodePredictor != nil && len(fx.CodePredictor.AcousticFrame) > 0
	cov.DecodedWAVSummary = fx.Decoder12Hz != nil
	cov.CompleteRuntimeTrace = cov.Prompt && cov.SemanticToken && cov.AcousticFrame && cov.DecodedWAVSummary
	if !cov.Prompt {
		cov.Missing = append(cov.Missing, "prompt")
	}
	if !cov.SemanticToken {
		cov.Missing = append(cov.Missing, "semantic_token")
	}
	if !cov.AcousticFrame {
		cov.Missing = append(cov.Missing, "acoustic_frame")
	}
	if !cov.DecodedWAVSummary {
		cov.Missing = append(cov.Missing, "decoded_wav_summary")
	}
	return cov
}

func (fx ReferenceFixture) Validate() error {
	if fx.Name == "" {
		return fmt.Errorf("qwen3tts fixture has empty name")
	}
	if fx.Variant != Base && fx.Variant != CustomVoice && fx.Variant != VoiceDesign {
		return fmt.Errorf("qwen3tts fixture %q has unknown variant %q", fx.Name, fx.Variant)
	}
	if fx.ModelSize == "" {
		return fmt.Errorf("qwen3tts fixture %q has empty model size", fx.Name)
	}
	if _, err := fx.Speaker.TokenID(); err != nil {
		return fmt.Errorf("qwen3tts fixture %q: %w", fx.Name, err)
	}
	if _, err := fx.Language.TokenID(); err != nil {
		return fmt.Errorf("qwen3tts fixture %q: %w", fx.Name, err)
	}
	if len(fx.Prompt.Text) == 0 || len(fx.Prompt.Codec) == 0 {
		return fmt.Errorf("qwen3tts fixture %q has empty prompt streams", fx.Name)
	}
	if fx.Talker != nil {
		if err := (SemanticTokenLayout{VocabSize: CodecVocabSize}).ValidateToken(fx.Talker.FirstSemanticToken); err != nil {
			return fmt.Errorf("qwen3tts fixture %q: %w", fx.Name, err)
		}
	}
	if fx.CodePredictor != nil && len(fx.CodePredictor.AcousticFrame) != 0 && len(fx.CodePredictor.AcousticFrame) != 15 {
		return fmt.Errorf("qwen3tts fixture %q acoustic frame length=%d, want 15", fx.Name, len(fx.CodePredictor.AcousticFrame))
	}
	if fx.Decoder12Hz != nil && (fx.Decoder12Hz.SampleRate <= 0 || fx.Decoder12Hz.Samples <= 0 || fx.Decoder12Hz.DurationS <= 0) {
		return fmt.Errorf("qwen3tts fixture %q has invalid decoder summary: %+v", fx.Name, fx.Decoder12Hz)
	}
	return nil
}
