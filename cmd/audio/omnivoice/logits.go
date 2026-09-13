package main

import (
	"context"
	"encoding/json"
	"fmt"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	model "github.com/rcarmo/go-pherence/models/omnivoice"
	"os"
	"runtime"
	"time"
)

// logits input is deliberately low-level: no implicit text tokenization or
// voice-codec encoding. It lets reference runtimes exchange exact token inputs.
type logitsInput struct {
	Tokens    int       `json:"tokens"`
	IDs       []int     `json:"ids"`
	AudioMask []bool    `json:"audio_mask"`
	Positions []int     `json:"positions"`
	Mask      []float32 `json:"mask"`
}

func runLogits(weights *loader.Weights, path string) error {
	if path == "" {
		return fmt.Errorf("logits mode requires -input JSON")
	}
	stat, err := os.Stat(path)
	if err != nil {
		return err
	}
	if stat.Size() > 16<<20 {
		return fmt.Errorf("input exceeds 16 MiB")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var input logitsInput
	if err = json.Unmarshal(raw, &input); err != nil {
		return err
	}
	if input.Tokens < 1 || input.Tokens > 256 {
		return fmt.Errorf("tokens must be 1..256")
	}
	started := time.Now()
	b, err := model.NewBackbone(weights, input.Tokens)
	if err != nil {
		return err
	}
	cfg := weights.Config
	out := make([]float32, cfg.NumAudioCodebook*input.Tokens*cfg.AudioVocabSize)
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	loaded := time.Now()
	if err = b.ForwardInto(context.Background(), out, input.IDs, input.AudioMask, input.Positions, input.Mask); err != nil {
		return err
	}
	elapsed := time.Since(loaded).Seconds()
	runtime.ReadMemStats(&after)
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"mode": "logits", "speech_generation": false, "shape": []int{cfg.NumAudioCodebook, input.Tokens, cfg.AudioVocabSize}, "logits": out, "setup_seconds": loaded.Sub(started).Seconds(), "forward_seconds": elapsed, "hot_allocations": after.Mallocs - before.Mallocs, "hot_allocated_bytes": after.TotalAlloc - before.TotalAlloc})
}
