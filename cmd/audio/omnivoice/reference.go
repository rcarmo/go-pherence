package main

import (
	"context"
	"encoding/json"
	"fmt"
	audio "github.com/rcarmo/go-264/audio"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	model "github.com/rcarmo/go-pherence/models/omnivoice"
	"io"
	"os"
	"path/filepath"
	"time"
)

func encodeReference(modelPath, wavePath, transcript string) (loader.CachedReferenceTokens, error) {
	wave, err := readReferenceAudio(wavePath)
	if err != nil {
		return loader.CachedReferenceTokens{}, err
	}
	encoder, err := model.LoadReferenceEncoder(filepath.Join(modelPath, "audio_tokenizer"))
	if err != nil {
		return loader.CachedReferenceTokens{}, err
	}
	return encoder.Encode(context.Background(), wave, transcript)
}

func readReferenceAudio(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}
	d, err := audio.Open(context.Background(), f, stat.Size(), audio.Options{TargetRate: 24000, TargetChannels: 1})
	if err != nil {
		return nil, err
	}
	defer d.Close()
	wave := make([]float32, 0, 5*24000)
	buf := make([]int16, 4096)
	for {
		n, _, err := d.ReadPCM(context.Background(), buf)
		if len(wave)+n > 20*24000 {
			return nil, fmt.Errorf("reference exceeds 20 seconds")
		}
		for _, v := range buf[:n] {
			wave = append(wave, float32(v)/32768)
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if n == 0 {
			return nil, io.ErrNoProgress
		}
	}
	if len(wave) < 2*24000 {
		return nil, fmt.Errorf("reference shorter than 2 seconds")
	}
	return wave, nil
}

func prepareRawPrompt(modelPath, wavePath, transcript, text string, frames int, language, instruct string, denoise bool) (loader.PreparedPrompt, error) {
	started := time.Now()
	ref, err := encodeReference(modelPath, wavePath, transcript)
	if err != nil {
		return loader.PreparedPrompt{}, err
	}
	fmt.Fprintf(os.Stderr, "native reference: %d frames in %.3fs\n", ref.Frames, time.Since(started).Seconds())
	cfg, err := loader.LoadConfig(modelPath)
	if err != nil {
		return loader.PreparedPrompt{}, err
	}
	tok, err := tokenizer.Load(filepath.Join(modelPath, "tokenizer.json"))
	if err != nil {
		return loader.PreparedPrompt{}, err
	}
	return loader.PrepareInferenceInputs(cfg, tok, text, frames, ref, loader.PreparePromptOptions{Language: language, Instruct: instruct, Denoise: denoise})
}

func writeReference(output string, ref loader.CachedReferenceTokens) error {
	if filepath.Ext(output) != ".json" {
		return fmt.Errorf("encode-reference requires new -output .json")
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
	if err = json.NewEncoder(f).Encode(ref); err != nil {
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	ok = true
	return nil
}
