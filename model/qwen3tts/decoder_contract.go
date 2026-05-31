package qwen3tts

import "fmt"

// Decoder12HzExecutionContract is a validation-only contract for the future
// CPU/reference Decoder12Hz implementation. It ties flattened acoustic-code
// input to mono PCM output sizing at 24kHz.
type Decoder12HzExecutionContract struct {
	Plan             RuntimeRequestPlan `json:"plan"`
	DecoderInput     DecoderInputLayout `json:"decoder_input"`
	Waveform         WaveformLayout     `json:"waveform"`
	MaxFrames        int                `json:"max_frames"`
	CodesPerFrame    int                `json:"codes_per_frame"`
	SamplesPerFrame  int                `json:"samples_per_frame"`
	MaxAcousticCodes int                `json:"max_acoustic_codes"`
	MaxSamples       int                `json:"max_samples"`
}

func NewDecoder12HzExecutionContract(plan RuntimeRequestPlan) (Decoder12HzExecutionContract, error) {
	if err := plan.Validate(); err != nil {
		return Decoder12HzExecutionContract{}, err
	}
	contract := Decoder12HzExecutionContract{Plan: plan, DecoderInput: plan.DecoderInput, Waveform: plan.Waveform, MaxFrames: plan.MaxFrames, CodesPerFrame: plan.DecoderInput.CodesPerFrame, SamplesPerFrame: plan.Waveform.SamplesPerFrame, MaxAcousticCodes: plan.MaxCodes, MaxSamples: plan.MaxSamples}
	return contract, contract.Validate()
}

func (c Decoder12HzExecutionContract) Validate() error {
	if err := c.Plan.Validate(); err != nil {
		return err
	}
	if err := c.DecoderInput.Validate(); err != nil {
		return err
	}
	if err := c.Waveform.Validate(); err != nil {
		return err
	}
	if c.MaxFrames <= 0 || c.MaxFrames != c.Plan.MaxFrames || c.CodesPerFrame != c.DecoderInput.CodesPerFrame || c.SamplesPerFrame != c.Waveform.SamplesPerFrame {
		return fmt.Errorf("invalid Qwen3-TTS Decoder12Hz contract limits: %+v", c)
	}
	wantCodes, err := c.DecoderInput.CodesForFrames(c.MaxFrames)
	if err != nil {
		return err
	}
	wantSamples, err := c.Waveform.SamplesForFrames(c.MaxFrames)
	if err != nil {
		return err
	}
	if c.MaxAcousticCodes != wantCodes || c.Plan.MaxCodes != wantCodes || c.MaxSamples != wantSamples || c.Plan.MaxSamples != wantSamples {
		return fmt.Errorf("invalid Qwen3-TTS Decoder12Hz sizing codes=%d/%d samples=%d/%d want_codes=%d want_samples=%d", c.MaxAcousticCodes, c.Plan.MaxCodes, c.MaxSamples, c.Plan.MaxSamples, wantCodes, wantSamples)
	}
	return nil
}

func (c Decoder12HzExecutionContract) ValidateInput(acoustic []uint32) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if len(acoustic) == 0 || len(acoustic) > c.MaxAcousticCodes {
		return fmt.Errorf("invalid Qwen3-TTS Decoder12Hz acoustic codes=%d max=%d", len(acoustic), c.MaxAcousticCodes)
	}
	return c.DecoderInput.ValidateCodes(acoustic)
}

func (c Decoder12HzExecutionContract) ValidateOutput(samples []float32) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if len(samples) == 0 || len(samples) > c.MaxSamples || len(samples)%c.SamplesPerFrame != 0 {
		return fmt.Errorf("invalid Qwen3-TTS Decoder12Hz samples=%d samples_per_frame=%d max=%d", len(samples), c.SamplesPerFrame, c.MaxSamples)
	}
	return nil
}
