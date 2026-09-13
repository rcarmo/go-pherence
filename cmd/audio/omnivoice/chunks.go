package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	model "github.com/rcarmo/go-pherence/models/omnivoice"
)

func runChunked(modelPath, mode, output, text, reference, transcript, cached, language, instruct string, maxFrames, steps int, denoise, preprocess, postprocess bool) error {
	started := time.Now()
	if strings.TrimSpace(text) == "" || maxFrames < 1 || maxFrames > 250 || steps < 1 || steps > 128 {
		return fmt.Errorf("chunk mode requires text, frames 1..250 and steps 1..128")
	}
	ext := ".wav"
	if mode == "plan-chunks" {
		ext = ".json"
	}
	if filepath.Ext(output) != ext {
		return fmt.Errorf("%s requires new -output %s", mode, ext)
	}
	if _, err := os.Lstat(output); !os.IsNotExist(err) {
		return fmt.Errorf("output exists or cannot be checked")
	}
	if (reference == "") == (cached == "") {
		return fmt.Errorf("choose exactly one raw or cached reference")
	}
	if reference != "" && strings.TrimSpace(transcript) == "" {
		return fmt.Errorf("raw reference requires transcript")
	}
	if preprocess && reference == "" {
		return fmt.Errorf("reference preprocessing requires raw audio")
	}
	var ref loader.CachedReferenceTokens
	var err error
	if reference != "" {
		ref, err = encodeReference(modelPath, reference, transcript, preprocess)
	} else {
		info, e := os.Stat(cached)
		if e != nil {
			return e
		}
		if info.Size() > 16<<20 {
			return fmt.Errorf("reference cache exceeds 16 MiB")
		}
		ref, err = loader.LoadCachedReferenceTokens(cached)
	}
	if err != nil {
		return err
	}
	cfg, err := loader.LoadConfig(modelPath)
	if err != nil {
		return err
	}
	tok, err := tokenizer.Load(filepath.Join(modelPath, "tokenizer.json"))
	if err != nil {
		return err
	}
	prompts, err := loader.PlanChunks(cfg, tok, text, ref, loader.PreparePromptOptions{Language: language, Instruct: instruct, Denoise: denoise}, maxFrames)
	if err != nil {
		return err
	}
	totalFrames := 0
	for _, p := range prompts {
		totalFrames += p.TargetFrames
	}
	if totalFrames > 15000 {
		return fmt.Errorf("chunk plan exceeds 10 minutes of generated audio")
	}
	if mode == "plan-chunks" {
		return writeChunkPlan(output, prompts)
	}
	weights, err := loader.OpenWeights(modelPath)
	if err != nil {
		return err
	}
	defer weights.Close()
	cw, err := loader.LoadCodecDecoder(filepath.Join(modelPath, "audio_tokenizer"))
	if err != nil {
		return err
	}
	decoder, err := model.NewCodecDecoder(cw)
	if err != nil {
		return err
	}
	// Keep one small backbone as arena owner; all per-chunk branches are siblings.
	owner, err := model.NewBackbone(weights, 1)
	if err != nil {
		return err
	}
	waves := make([][]float32, 0, len(prompts))
	timings := make([]float64, 0, len(prompts))
	for i, p := range prompts {
		at := time.Now()
		wave, e := generateChunk(owner, decoder, p, steps, postprocess)
		if e != nil {
			return fmt.Errorf("chunk %d: %w", i+1, e)
		}
		waves = append(waves, wave)
		timings = append(timings, time.Since(at).Seconds())
		fmt.Fprintf(os.Stderr, "chunk %d/%d: %d frames, %.3fs\n", i+1, len(prompts), p.TargetFrames, timings[i])
	}
	// Explicit native boundary policy: 5ms edge fade per chunk and 100ms silence.
	joined, err := joinChunkWaves(waves, 120, 2400)
	if err != nil {
		return err
	}
	gain, err := loader.WriteSyntheticWAV(output, joined, 24000)
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"mode": mode, "synthetic": true, "native_inference": true, "reference_encoding_native": reference != "", "silence_preprocessing": preprocess, "output_postprocessing": postprocess, "chunks": len(prompts), "chunk_texts": chunkTexts(prompts), "target_frames": totalFrames, "chunk_seconds": timings, "audio_seconds": float64(len(joined)) / 24000, "command_seconds": time.Since(started).Seconds(), "steps": steps, "seed_per_chunk": 42, "boundary_fade_ms": 5, "boundary_gap_ms": 100, "gain": gain, "output": output})
}

func generateChunk(owner *model.Backbone, decoder *model.CodecDecoder, p loader.PreparedPrompt, steps int, postprocess bool) ([]float32, error) {
	cond, err := model.NewBackboneSibling(owner, p.Conditional.Tokens)
	if err != nil {
		return nil, err
	}
	uncond, err := model.NewBackboneSibling(owner, p.Unconditional.Tokens)
	if err != nil {
		return nil, err
	}
	cfg := model.DefaultGenerationConfig()
	cfg.Steps = steps
	g, err := model.NewGeneration(cond, uncond, p.TargetFrames, cfg)
	if err != nil {
		return nil, err
	}
	codes := make([]int, 8*p.TargetFrames)
	if err = g.GenerateInto(context.Background(), codes, p.Conditional.IDs, p.Conditional.AudioMask, p.Unconditional.IDs, p.Unconditional.AudioMask); err != nil {
		return nil, err
	}
	wave, err := decoder.Decode(context.Background(), codes, 8, p.TargetFrames)
	if err != nil {
		return nil, err
	}
	if postprocess {
		o := loader.DefaultSilenceOptions()
		o.MidSilenceMS = 500
		o.LeadingKeepMS = 100
		o.TrailingKeepMS = 100
		wave, err = loader.RemoveSilenceMono24k(wave, o)
		if err != nil {
			return nil, err
		}
	}
	if p.RefRMS != nil && *p.RefRMS < .1 {
		gain := float32(*p.RefRMS / .1)
		for i := range wave {
			wave[i] *= gain
		}
	}
	if len(wave) == 0 {
		return nil, fmt.Errorf("empty generated chunk")
	}
	return wave, nil
}

func joinChunkWaves(waves [][]float32, fade, gap int) ([]float32, error) {
	if len(waves) == 0 || len(waves) > 128 || fade < 0 || fade > 2400 || gap < 0 || gap > 24000 {
		return nil, fmt.Errorf("invalid chunk assembly")
	}
	n := (len(waves) - 1) * gap
	for _, w := range waves {
		if len(w) == 0 || len(w) > 250*960 {
			return nil, fmt.Errorf("invalid chunk length")
		}
		n += len(w)
		if n > 610*24000 {
			return nil, fmt.Errorf("assembled audio exceeds limit")
		}
		for _, v := range w {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("nonfinite chunk")
			}
		}
	}
	out := make([]float32, n)
	at := 0
	for _, w := range waves {
		copy(out[at:], w)
		f := min(fade, len(w)/2)
		if f > 1 {
			for i := 0; i < f; i++ {
				gain := float32(i) / float32(f-1)
				out[at+i] *= gain
				out[at+len(w)-1-i] *= gain
			}
		}
		at += len(w) + gap
	}
	return out, nil
}

func chunkTexts(p []loader.PreparedPrompt) []string {
	out := make([]string, len(p))
	for i := range p {
		out[i] = p[i].Text
	}
	return out
}

func writeChunkPlan(path string, p []loader.PreparedPrompt) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	ok := false
	defer func() {
		f.Close()
		if !ok {
			os.Remove(path)
		}
	}()
	if err = json.NewEncoder(f).Encode(map[string]any{"mode": "plan-chunks", "chunks": p, "boundary_policy": "native 5ms fades + 100ms gap; not upstream chunk-equivalent"}); err != nil {
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	ok = true
	return nil
}
