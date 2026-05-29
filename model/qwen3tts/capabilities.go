package qwen3tts

import "fmt"

// ConditioningMode describes the external inputs required by a Qwen3-TTS
// variant. Runtime code should validate these before attempting synthesis.
type ConditioningMode string

const (
	ConditioningCustomVoice    ConditioningMode = "custom_voice"
	ConditioningReferenceAudio ConditioningMode = "reference_audio"
	ConditioningVoiceDesign    ConditioningMode = "voice_design"
)

type Capabilities struct {
	ModelType           ModelType        `json:"model_type"`
	Conditioning        ConditioningMode `json:"conditioning"`
	RequiresSpeaker     bool             `json:"requires_speaker"`
	RequiresLanguage    bool             `json:"requires_language"`
	RequiresAudio       bool             `json:"requires_audio"`
	RequiresVoicePrompt bool             `json:"requires_voice_prompt"`
	HasSpeakerEncoder   bool             `json:"has_speaker_encoder"`
	SupportedSpeakers   []Speaker        `json:"supported_speakers,omitempty"`
	SupportedLanguages  []Language       `json:"supported_languages,omitempty"`
}

// Capabilities reports the conditioning contract implied by the checkpoint
// variant. It is intentionally conservative: Base requires reference audio even
// if a speaker encoder is absent from metadata, because no Base runtime path is
// valid without reference conditioning.
func (c ParsedConfig) Capabilities() (Capabilities, error) {
	caps := Capabilities{ModelType: c.ModelType, HasSpeakerEncoder: c.SpeakerEncoder != nil}
	switch c.ModelType {
	case CustomVoice:
		caps.Conditioning = ConditioningCustomVoice
		caps.RequiresSpeaker = true
		caps.RequiresLanguage = true
		caps.SupportedSpeakers = AllSpeakers()
		caps.SupportedLanguages = AllLanguages()
	case Base:
		caps.Conditioning = ConditioningReferenceAudio
		caps.RequiresAudio = true
		caps.RequiresLanguage = true
		caps.SupportedLanguages = AllLanguages()
	case VoiceDesign:
		caps.Conditioning = ConditioningVoiceDesign
		caps.RequiresVoicePrompt = true
		caps.RequiresLanguage = true
		caps.SupportedLanguages = AllLanguages()
	default:
		return Capabilities{}, fmt.Errorf("unknown Qwen3-TTS model type %q", c.ModelType)
	}
	return caps, nil
}

func (c ParsedConfig) ValidateConditioning(req ConditioningRequest) error {
	caps, err := c.Capabilities()
	if err != nil {
		return err
	}
	if caps.RequiresLanguage {
		if _, err := req.Language.TokenID(); err != nil {
			return err
		}
	}
	if caps.RequiresSpeaker {
		if _, err := req.Speaker.TokenID(); err != nil {
			return err
		}
	}
	if caps.RequiresAudio && !req.HasReferenceAudio {
		return fmt.Errorf("Qwen3-TTS %s requires reference audio conditioning", c.ModelType)
	}
	if caps.RequiresVoicePrompt && req.VoicePrompt == "" {
		return fmt.Errorf("Qwen3-TTS %s requires voice design prompt conditioning", c.ModelType)
	}
	if !caps.RequiresSpeaker && req.Speaker != "" {
		return fmt.Errorf("Qwen3-TTS %s does not accept fixed CustomVoice speaker tokens", c.ModelType)
	}
	if !caps.RequiresAudio && req.HasReferenceAudio {
		return fmt.Errorf("Qwen3-TTS %s does not accept reference audio conditioning", c.ModelType)
	}
	if !caps.RequiresVoicePrompt && req.VoicePrompt != "" {
		return fmt.Errorf("Qwen3-TTS %s does not accept voice design prompt conditioning", c.ModelType)
	}
	return nil
}

type ConditioningRequest struct {
	Language          Language `json:"language"`
	Speaker           Speaker  `json:"speaker,omitempty"`
	HasReferenceAudio bool     `json:"has_reference_audio,omitempty"`
	ReferenceAudio    string   `json:"reference_audio,omitempty"`
	VoicePrompt       string   `json:"voice_prompt,omitempty"`
}

type ConditioningValidation struct {
	Valid   bool                `json:"valid"`
	Request ConditioningRequest `json:"request"`
	Error   string              `json:"error,omitempty"`
}

func (c ParsedConfig) CheckConditioning(req ConditioningRequest) ConditioningValidation {
	if req.ReferenceAudio != "" {
		req.HasReferenceAudio = true
	}
	if err := c.ValidateConditioning(req); err != nil {
		return ConditioningValidation{Valid: false, Request: req, Error: err.Error()}
	}
	return ConditioningValidation{Valid: true, Request: req}
}
