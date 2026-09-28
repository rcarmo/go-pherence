package qwen3tts

import "fmt"

// Decoder12HzExecutionContract ties the CodePredictor's 15 acoustic groups and
// Talker's semantic group to complete 16-codebook frames and exact 24kHz PCM
// output sizing.
type Decoder12HzExecutionContract struct {
	Plan             RuntimeRequestPlan `json:"plan"`
	DecoderInput     DecoderInputLayout `json:"decoder_input"`
	Waveform         WaveformLayout     `json:"waveform"`
	MaxFrames        int                `json:"max_frames"`
	CodesPerFrame    int                `json:"codes_per_frame"`
	AcousticPerFrame int                `json:"acoustic_per_frame"`
	SamplesPerFrame  int                `json:"samples_per_frame"`
	MaxDecoderCodes  int                `json:"max_decoder_codes"`
	MaxAcousticCodes int                `json:"max_acoustic_codes"`
	MaxSamples       int                `json:"max_samples"`
}

func NewDecoder12HzExecutionContract(plan RuntimeRequestPlan) (Decoder12HzExecutionContract, error) {
	if err := plan.Validate(); err != nil {
		return Decoder12HzExecutionContract{}, err
	}
	maxDecoderCodes, err := plan.DecoderInput.CodesForFrames(plan.MaxFrames)
	if err != nil {
		return Decoder12HzExecutionContract{}, err
	}
	contract := Decoder12HzExecutionContract{
		Plan: plan, DecoderInput: plan.DecoderInput, Waveform: plan.Waveform,
		MaxFrames: plan.MaxFrames, CodesPerFrame: plan.DecoderInput.CodesPerFrame,
		AcousticPerFrame: plan.DecoderInput.AcousticGroups, SamplesPerFrame: plan.Waveform.SamplesPerFrame,
		MaxDecoderCodes: maxDecoderCodes, MaxAcousticCodes: plan.MaxCodes, MaxSamples: plan.MaxSamples,
	}
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
	if c.MaxFrames <= 0 || c.MaxFrames != c.Plan.MaxFrames || c.CodesPerFrame != c.DecoderInput.CodesPerFrame || c.AcousticPerFrame != c.DecoderInput.AcousticGroups || c.SamplesPerFrame != c.Waveform.SamplesPerFrame {
		return fmt.Errorf("invalid Qwen3-TTS Decoder12Hz contract limits: %+v", c)
	}
	wantDecoder, err := c.DecoderInput.CodesForFrames(c.MaxFrames)
	if err != nil {
		return err
	}
	wantAcoustic, err := c.DecoderInput.AcousticCodesForFrames(c.MaxFrames)
	if err != nil {
		return err
	}
	wantSamples, err := c.Waveform.SamplesForFrames(c.MaxFrames)
	if err != nil {
		return err
	}
	if c.MaxDecoderCodes != wantDecoder || c.MaxAcousticCodes != wantAcoustic || c.Plan.MaxCodes != wantAcoustic || c.MaxSamples != wantSamples || c.Plan.MaxSamples != wantSamples {
		return fmt.Errorf("invalid Qwen3-TTS Decoder12Hz sizing decoder=%d/%d acoustic=%d/%d samples=%d/%d", c.MaxDecoderCodes, wantDecoder, c.MaxAcousticCodes, wantAcoustic, c.MaxSamples, wantSamples)
	}
	return nil
}

// JoinInput validates staged outputs and constructs frame-major
// [semantic, acoustic_0..14] decoder codes.
func (c Decoder12HzExecutionContract) JoinInput(semantic, acoustic []uint32) ([]uint32, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	if len(semantic) == 0 || len(semantic) > c.MaxFrames || len(acoustic) > c.MaxAcousticCodes {
		return nil, fmt.Errorf("invalid Qwen3-TTS Decoder12Hz staged input semantic=%d acoustic=%d", len(semantic), len(acoustic))
	}
	return c.DecoderInput.JoinFrames(semantic, acoustic)
}

func (c Decoder12HzExecutionContract) ValidateInput(codes []uint32) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if len(codes) == 0 || len(codes) > c.MaxDecoderCodes {
		return fmt.Errorf("invalid Qwen3-TTS Decoder12Hz codes=%d max=%d", len(codes), c.MaxDecoderCodes)
	}
	return c.DecoderInput.ValidateCodes(codes)
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

// ValidateOutputForFrames additionally binds output geometry to the staged
// semantic/acoustic frame count rather than merely accepting any whole number
// of frames below the request maximum.
func (c Decoder12HzExecutionContract) ValidateOutputForFrames(samples []float32, frames int) error {
	if err := c.ValidateOutput(samples); err != nil {
		return err
	}
	want, err := c.Waveform.SamplesForFrames(frames)
	if err != nil {
		return err
	}
	if frames > c.MaxFrames || len(samples) != want {
		return fmt.Errorf("invalid Qwen3-TTS Decoder12Hz samples=%d want=%d frames=%d", len(samples), want, frames)
	}
	return nil
}
