package qwen3tts

import "fmt"

// RuntimeRequest captures the minimal inputs and limits a future Qwen3-TTS
// runtime needs before executing Talker, CodePredictor, and Decoder12Hz. It is
// validation-only: no audio generation is performed here.
type RuntimeRequest struct {
	Conditioning ConditioningRequest `json:"conditioning"`
	Prompt       PromptIDs           `json:"prompt"`
	MaxFrames    int                 `json:"max_frames"`
	MaxSeconds   float64             `json:"max_seconds"`
}

type RuntimeRequestPlan struct {
	Conditioning ConditioningValidation `json:"conditioning"`
	PromptLayout PromptRuntimeLayout    `json:"prompt_layout"`
	DecoderInput DecoderInputLayout     `json:"decoder_input"`
	Waveform     WaveformLayout         `json:"waveform"`
	MaxFrames    int                    `json:"max_frames"`
	MaxSamples   int                    `json:"max_samples"`
	MaxCodes     int                    `json:"max_codes"`
}

func NewRuntimeRequestPlan(cfg ParsedConfig, req RuntimeRequest) (RuntimeRequestPlan, error) {
	if err := cfg.Validate(); err != nil {
		return RuntimeRequestPlan{}, err
	}
	conditioning := cfg.CheckConditioning(req.Conditioning)
	if !conditioning.Valid {
		return RuntimeRequestPlan{}, fmt.Errorf("invalid Qwen3-TTS conditioning: %s", conditioning.Error)
	}
	promptLayout, err := NewPromptRuntimeLayout(cfg, req.Prompt)
	if err != nil {
		return RuntimeRequestPlan{}, err
	}
	decoderInput, err := NewDecoderInputLayout(cfg)
	if err != nil {
		return RuntimeRequestPlan{}, err
	}
	decoderPlan, err := decoderInput.DecoderPlan()
	if err != nil {
		return RuntimeRequestPlan{}, err
	}
	waveform, err := NewWaveformLayout(decoderPlan)
	if err != nil {
		return RuntimeRequestPlan{}, err
	}
	maxFrames := req.MaxFrames
	if maxFrames == 0 && req.MaxSeconds > 0 {
		maxFrames = int(req.MaxSeconds * float64(decoderInput.FrameRateHz))
	}
	if maxFrames <= 0 {
		return RuntimeRequestPlan{}, fmt.Errorf("invalid Qwen3-TTS max frames=%d max_seconds=%g", req.MaxFrames, req.MaxSeconds)
	}
	maxSamples, err := waveform.SamplesForFrames(maxFrames)
	if err != nil {
		return RuntimeRequestPlan{}, err
	}
	maxCodes, err := decoderInput.CodesForFrames(maxFrames)
	if err != nil {
		return RuntimeRequestPlan{}, err
	}
	plan := RuntimeRequestPlan{Conditioning: conditioning, PromptLayout: promptLayout, DecoderInput: decoderInput, Waveform: waveform, MaxFrames: maxFrames, MaxSamples: maxSamples, MaxCodes: maxCodes}
	return plan, plan.Validate()
}

func (p RuntimeRequestPlan) Validate() error {
	if !p.Conditioning.Valid {
		return fmt.Errorf("invalid Qwen3-TTS request conditioning: %s", p.Conditioning.Error)
	}
	if err := p.PromptLayout.Validate(); err != nil {
		return err
	}
	if err := p.DecoderInput.Validate(); err != nil {
		return err
	}
	if err := p.Waveform.Validate(); err != nil {
		return err
	}
	if p.MaxFrames <= 0 || p.MaxSamples <= 0 || p.MaxCodes <= 0 {
		return fmt.Errorf("invalid Qwen3-TTS request limits: %+v", p)
	}
	wantSamples, err := p.Waveform.SamplesForFrames(p.MaxFrames)
	if err != nil {
		return err
	}
	if p.MaxSamples != wantSamples {
		return fmt.Errorf("invalid Qwen3-TTS request samples=%d want=%d", p.MaxSamples, wantSamples)
	}
	wantCodes, err := p.DecoderInput.CodesForFrames(p.MaxFrames)
	if err != nil {
		return err
	}
	if p.MaxCodes != wantCodes {
		return fmt.Errorf("invalid Qwen3-TTS request codes=%d want=%d", p.MaxCodes, wantCodes)
	}
	return nil
}
