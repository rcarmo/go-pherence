package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

// prepareCachedPrompt never starts Python or performs reference audio encoding.
func prepareCachedPrompt(modelPath, referencePath, text string, frames int, language, instruct string, denoise bool) (loader.PreparedPrompt, error) {
	if referencePath == "" {
		return loader.PreparedPrompt{}, fmt.Errorf("cached reference JSON is required")
	}
	cfg, err := loader.LoadConfig(modelPath)
	if err != nil {
		return loader.PreparedPrompt{}, err
	}
	info, err := os.Stat(referencePath)
	if err != nil {
		return loader.PreparedPrompt{}, err
	}
	if info.Size() > 16<<20 {
		return loader.PreparedPrompt{}, fmt.Errorf("reference cache exceeds 16 MiB")
	}
	ref, err := loader.LoadCachedReferenceTokens(referencePath)
	if err != nil {
		return loader.PreparedPrompt{}, err
	}
	tok, err := tokenizer.Load(filepath.Join(modelPath, "tokenizer.json"))
	if err != nil {
		return loader.PreparedPrompt{}, err
	}
	return loader.PrepareInferenceInputs(cfg, tok, text, frames, ref, loader.PreparePromptOptions{Language: language, Instruct: instruct, Denoise: denoise})
}

func writePreparedPrompt(output string, p loader.PreparedPrompt) error {
	if strings.ToLower(filepath.Ext(output)) != ".json" {
		return fmt.Errorf("prepare requires a new -output .json")
	}
	f, err := os.OpenFile(output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	ok := false
	defer func() {
		f.Close()
		if !ok {
			os.Remove(output)
		}
	}()
	if err = json.NewEncoder(f).Encode(p); err != nil {
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	ok = true
	return nil
}

func generationPrompt(p loader.PreparedPrompt) preparedPrompt {
	return preparedPrompt{
		Conditional:   logitsInput{Tokens: p.Conditional.Tokens, IDs: p.Conditional.IDs, AudioMask: p.Conditional.AudioMask},
		Unconditional: logitsInput{Tokens: p.Unconditional.Tokens, IDs: p.Unconditional.IDs, AudioMask: p.Unconditional.AudioMask},
		Target:        p.TargetFrames, Text: p.Text, RefRMS: p.RefRMS,
	}
}

// Validate cheap input/output constraints before model loading or reference encoding.
func validatePreparationFlags(mode, input, output, text, reference, transcript, cached string, frames, steps int) error {
	if mode != "prepare" && mode != "synthesize" && mode != "encode-reference" {
		return nil
	}
	ext := ".json"
	if mode == "synthesize" {
		ext = ".wav"
	}
	if output == "" || filepath.Ext(output) != ext {
		return fmt.Errorf("%s requires a new -output %s", mode, ext)
	}
	if _, err := os.Lstat(output); !os.IsNotExist(err) {
		return fmt.Errorf("output exists or cannot be checked")
	}
	if input != "" {
		return fmt.Errorf("%s does not accept -input", mode)
	}
	if mode == "encode-reference" {
		if reference == "" || strings.TrimSpace(transcript) == "" || cached != "" {
			return fmt.Errorf("encode-reference requires reference and transcript, without cached tokens")
		}
		return nil
	}
	if strings.TrimSpace(text) == "" || frames < 1 || frames > 250 {
		return fmt.Errorf("target text and frames 1..250 required")
	}
	if mode == "synthesize" && (steps < 1 || steps > 128) {
		return fmt.Errorf("steps must be 1..128")
	}
	if (reference == "") == (cached == "") {
		return fmt.Errorf("choose exactly one of -reference and -reference-tokens")
	}
	if reference != "" && strings.TrimSpace(transcript) == "" {
		return fmt.Errorf("raw reference requires transcript")
	}
	return nil
}
