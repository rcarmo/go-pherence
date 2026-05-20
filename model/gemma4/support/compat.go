//go:build diagnostic
// +build diagnostic

package model

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
