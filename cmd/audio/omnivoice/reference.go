package main

import (
	"context"
	"encoding/json"
	"fmt"
	audio "github.com/rcarmo/go-264/audio"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	model "github.com/rcarmo/go-pherence/model/omnivoice"
	"io"
	"math"
	"os"
	"path/filepath"
	"time"
)

func encodeReference(modelPath, wavePath, transcript string, preprocess bool) (loader.CachedReferenceTokens, error) {
	wave, err := readReferenceAudio(wavePath)
	if err != nil {
		return loader.CachedReferenceTokens{}, err
	}
	var rms float64
	if preprocess {
		wave, rms, err = preprocessReference(wave)
		if err != nil {
			return loader.CachedReferenceTokens{}, err
		}
	}
	encoder, err := model.LoadReferenceEncoder(filepath.Join(modelPath, "audio_tokenizer"))
	if err != nil {
		return loader.CachedReferenceTokens{}, err
	}
	if preprocess {
		return encoder.EncodePreprocessed(context.Background(), wave, transcript, rms)
	}
	return encoder.Encode(context.Background(), wave, transcript)
}

func preprocessReference(wave []float32) ([]float32, float64, error) {
	var sum float64
	for _, v := range wave {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, 0, fmt.Errorf("nonfinite reference")
		}
		sum += float64(v) * float64(v)
	}
	if len(wave) == 0 {
		return nil, 0, fmt.Errorf("empty reference")
	}
	rms := math.Sqrt(sum / float64(len(wave)))
	if rms < 1e-7 {
		return nil, 0, fmt.Errorf("silent reference")
	}
	normalized := append([]float32(nil), wave...)
	if rms < .1 {
		gain := float32(.1 / rms)
		for i := range normalized {
			normalized[i] *= gain
		}
	}
	trimmed, err := loader.RemoveReferenceSilenceMono24k(normalized)
	return trimmed, rms, err
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

func prepareRawPrompt(modelPath, wavePath, transcript, text string, frames int, language, instruct string, denoise, preprocess bool) (loader.PreparedPrompt, error) {
	started := time.Now()
	ref, err := encodeReference(modelPath, wavePath, transcript, preprocess)
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
