package main

import (
	"context"
	"encoding/json"
	"fmt"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	model "github.com/rcarmo/go-pherence/models/omnivoice"
	"math"
	"os"
	"path/filepath"
	"time"
)

// Prepared prompt inputs may be exported by a reference runtime. This command
// performs all iterative inference and waveform decoding natively; prompt
// tokenization/reference encoding are explicitly outside this input contract.
type preparedPrompt struct {
	CommandStarted  time.Time   `json:"-"`
	NativeReference bool        `json:"-"`
	Conditional     logitsInput `json:"conditional"`
	Unconditional   logitsInput `json:"unconditional"`
	Target          int         `json:"target_frames"`
	Text            string      `json:"text"`
	Reference       string      `json:"reference"`
	RefRMS          *float64    `json:"ref_rms,omitempty"`
}

func runGenerate(weights *loader.Weights, input, output, codecPath string, steps int) error {
	if input == "" || output == "" || filepath.Ext(output) != ".wav" {
		return fmt.Errorf("generate requires -input prepared JSON and new -output .wav")
	}
	if _, err := os.Lstat(output); !os.IsNotExist(err) {
		return fmt.Errorf("output exists or cannot be checked")
	}
	stat, err := os.Stat(input)
	if err != nil {
		return err
	}
	if stat.Size() > 16<<20 {
		return fmt.Errorf("prepared input exceeds 16 MiB")
	}
	raw, err := os.ReadFile(input)
	if err != nil {
		return err
	}
	var p preparedPrompt
	if err = json.Unmarshal(raw, &p); err != nil {
		return err
	}
	return generatePrompt(weights, p, output, codecPath, steps, true)
}

func generatePrompt(weights *loader.Weights, p preparedPrompt, output, codecPath string, steps int, prepared bool) error {
	if output == "" || filepath.Ext(output) != ".wav" {
		return fmt.Errorf("new -output .wav required")
	}
	if _, err := os.Lstat(output); !os.IsNotExist(err) {
		return fmt.Errorf("output exists or cannot be checked")
	}
	if p.RefRMS != nil && (math.IsNaN(*p.RefRMS) || math.IsInf(*p.RefRMS, 0) || *p.RefRMS < 0) {
		return fmt.Errorf("invalid reference RMS")
	}
	if p.Conditional.Tokens < 1 || p.Conditional.Tokens > 512 || p.Unconditional.Tokens < 1 || p.Unconditional.Tokens > 512 {
		return fmt.Errorf("prompt token limit is 512")
	}
	if p.Target < 1 || p.Target > 250 {
		return fmt.Errorf("target_frames must be 1..250")
	}
	if len(p.Conditional.Positions) > 0 || len(p.Unconditional.Positions) > 0 || len(p.Conditional.Mask) > 0 || len(p.Unconditional.Mask) > 0 {
		return fmt.Errorf("generate requires implicit positions and full attention")
	}
	started := time.Now()
	cond, err := model.NewBackbone(weights, p.Conditional.Tokens)
	if err != nil {
		return err
	}
	uncond, err := model.NewBackbone(weights, p.Unconditional.Tokens)
	if err != nil {
		return err
	}
	cfg := model.DefaultGenerationConfig()
	cfg.Steps = steps
	generation, err := model.NewGeneration(cond, uncond, p.Target, cfg)
	if err != nil {
		return err
	}
	codes := make([]int, weights.Config.NumAudioCodebook*p.Target)
	if err = generation.GenerateInto(context.Background(), codes, p.Conditional.IDs, p.Conditional.AudioMask, p.Unconditional.IDs, p.Unconditional.AudioMask); err != nil {
		return err
	}
	generated := time.Now()
	codec, err := loader.LoadCodecDecoder(codecPath)
	if err != nil {
		return err
	}
	decoder, err := model.NewCodecDecoder(codec)
	if err != nil {
		return err
	}
	wave, err := decoder.Decode(context.Background(), codes, weights.Config.NumAudioCodebook, p.Target)
	if err != nil {
		return err
	}
	// Match upstream loudness restoration after reference normalization.
	referenceGain := float32(1)
	if p.RefRMS != nil && *p.RefRMS < .1 {
		referenceGain = float32(*p.RefRMS / .1)
	}
	if referenceGain != 1 {
		for i := range wave {
			wave[i] *= referenceGain
		}
	}
	gain, err := loader.WriteSyntheticWAV(output, wave, codec.SampleRate)
	if err != nil {
		return err
	}
	commandSeconds := time.Since(started).Seconds()
	if !p.CommandStarted.IsZero() {
		commandSeconds = time.Since(p.CommandStarted).Seconds()
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"command_seconds": commandSeconds, "silence_preprocessing": false, "mode": "generate", "synthetic": true, "prepared_prompt_required": prepared, "reference_encoding_native": p.NativeReference, "reference_gain": referenceGain, "native_inference": true, "text": p.Text, "reference": p.Reference, "steps": steps, "seed": cfg.Seed, "rng": "Go PCG, not PyTorch RNG parity", "audio_seconds": float64(len(wave)) / float64(codec.SampleRate), "sample_rate": codec.SampleRate, "generation_seconds": generated.Sub(started).Seconds(), "decode_and_save_seconds": time.Since(generated).Seconds(), "total_seconds": time.Since(started).Seconds(), "gain": gain, "output": output})
}
