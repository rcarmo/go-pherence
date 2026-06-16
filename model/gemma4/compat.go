//go:build diagnostic
// +build diagnostic

package model

import (
	"math"
	"os"
	"path/filepath"
)

import basemodel "github.com/rcarmo/go-pherence/model"
import simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
import _ "unsafe"

type LlamaModel = basemodel.LlamaModel
type GPUModel = basemodel.GPUModel

var LoadLlama = basemodel.LoadLlama
var LoadGPUModel = basemodel.LoadGPUModel
var LoadGPUModelWithLayers = basemodel.LoadGPUModelWithLayers

var mcuda = &LlamaModel{}

// ForceOnTheFly is kept for legacy diagnostics that predate the current loader
// path. The production model package no longer consumes this toggle directly.
var ForceOnTheFly bool

func bf16Slice(x []float32) { simd.ToBF16(x) }

func gemma4Path() string {
	if p := os.Getenv("GEMMA4_PATH"); p != "" {
		return p
	}
	root := "."
	if wd, err := os.Getwd(); err == nil {
		for dir := wd; ; dir = filepath.Dir(dir) {
			if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
				root = dir
				break
			}
			if parent := filepath.Dir(dir); parent == dir {
				break
			}
		}
	}
	for _, rel := range []string{"models/gemma4-e2b-mlx4", "models/gemma4-e2b-it", "models/gemma4-e4b-it-4bit", "models/gemma4-31b-it-4bit"} {
		p := filepath.Join(root, rel)
		if _, err := os.Stat(filepath.Join(p, "config.json")); err == nil {
			return p
		}
	}
	return filepath.Join(root, "models/gemma4-31b-it-4bit")
}

func wrapGemma4PromptForTest(m *LlamaModel, prompt string) []int {
	ids := m.Tok.Encode(prompt)
	cfg := m.Config
	turnStart, turnEnd := -1, -1
	newlineID := -1
	for id, tok := range m.Tok.InvVocab {
		if tok == "<|turn>" {
			turnStart = id
		}
		if tok == "<turn|>" {
			turnEnd = id
		}
		if tok == "\n" {
			newlineID = id
		}
	}
	if cfg.BOSTokenID > 0 && turnStart >= 0 && turnEnd >= 0 && newlineID >= 0 {
		user := m.Tok.Encode("user")
		mdl := m.Tok.Encode("model")
		wrapped := []int{cfg.BOSTokenID, turnStart}
		wrapped = append(wrapped, user...)
		wrapped = append(wrapped, newlineID)
		wrapped = append(wrapped, ids...)
		wrapped = append(wrapped, turnEnd, newlineID, turnStart)
		wrapped = append(wrapped, mdl...)
		wrapped = append(wrapped, newlineID)
		return wrapped
	}
	return ids
}

func diffStats(a, b []float32) (maxAbs, meanAbs float64) {
	if len(a) != len(b) {
		return math.Inf(1), math.Inf(1)
	}
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		if d > maxAbs {
			maxAbs = d
		}
		meanAbs += d
	}
	if len(a) > 0 {
		meanAbs /= float64(len(a))
	}
	return maxAbs, meanAbs
}

//go:linkname debugLayerHook github.com/rcarmo/go-pherence/model.debugLayerHook
var debugLayerHook func(backend string, step, layer int, hidden []float32)

//go:linkname debugLogitsHook github.com/rcarmo/go-pherence/model.debugLogitsHook
var debugLogitsHook func(backend string, step int, hidden, logits []float32)

//go:linkname debugOpHook github.com/rcarmo/go-pherence/model.debugOpHook
var debugOpHook func(backend string, step, layer int, op string, vec []float32)

//go:linkname debugCPUHiddenInOverrideHook github.com/rcarmo/go-pherence/model.debugCPUHiddenInOverrideHook
var debugCPUHiddenInOverrideHook func(step, layer int, hidden []float32)

//go:linkname debugCPUPerLayerInputsOverrideHook github.com/rcarmo/go-pherence/model.debugCPUPerLayerInputsOverrideHook
var debugCPUPerLayerInputsOverrideHook func(step int, perLayerInputs [][]float32)

//go:linkname debugCPUMLPInputOverrideHook github.com/rcarmo/go-pherence/model.debugCPUMLPInputOverrideHook
var debugCPUMLPInputOverrideHook func(step, layer int, mlpInput []float32)
