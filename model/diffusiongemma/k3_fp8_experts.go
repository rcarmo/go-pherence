package diffusiongemma

import (
	"fmt"
	"strings"
)

// Per-expert FP8 checkpoints store experts.{N}.{gate,up,down}_proj.weight plus
// matching .weight_scale tensors instead of fused 3D expert tensors.
func perExpertTensorName(layerPrefix string, expertID int, proj string) string {
	return fmt.Sprintf("%s.experts.%d.%s.weight", layerPrefix, expertID, proj)
}

func perExpertScaleName(layerPrefix string, expertID int, proj string) string {
	return fmt.Sprintf("%s.experts.%d.%s.weight_scale", layerPrefix, expertID, proj)
}

func layerPrefixFromBinding(binding *TensorBinding) string {
	if binding == nil {
		return ""
	}
	name := binding.Name
	for _, marker := range []string{".router.", ".mlp.", ".self_attn.", ".input_layernorm"} {
		if idx := strings.Index(name, marker); idx > 0 {
			return name[:idx]
		}
	}
	return ""
}

// loadPerExpertFP8 loads one per-expert projection as dequantized F32.
func loadPerExpertFP8(weights *TextWeights, layerPrefix string, expertID int, proj string) ([]float32, int, int, error) {
	wName := perExpertTensorName(layerPrefix, expertID, proj)
	raw, dtype, shape, err := weights.RawTensor(wName)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("per-expert %s: %w", wName, err)
	}
	if len(shape) != 2 || shape[0] <= 0 || shape[1] <= 0 {
		return nil, 0, 0, fmt.Errorf("per-expert %s shape %v want 2D", wName, shape)
	}
	rows, cols := shape[0], shape[1]
	out := make([]float32, rows*cols)
	if dtype != "F8_E4M3" && dtype != "F8_E4M3FN" {
		if err := decodeFloatRowTo(out, raw, dtype); err != nil {
			return nil, 0, 0, err
		}
		return out, rows, cols, nil
	}
	sName := perExpertScaleName(layerPrefix, expertID, proj)
	sRaw, sDtype, sShape, err := weights.RawTensor(sName)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("per-expert scale %s: %w", sName, err)
	}
	nScale := 1
	for _, d := range sShape {
		if d <= 0 {
			return nil, 0, 0, fmt.Errorf("per-expert scale %s invalid shape %v", sName, sShape)
		}
		nScale *= d
	}
	if nScale != 1 && nScale != rows {
		return nil, 0, 0, fmt.Errorf("per-expert scale %s shape %v gives %d values, want 1 or %d", sName, sShape, nScale, rows)
	}
	scales := make([]float32, nScale)
	if err := decodeFloatRowTo(scales, sRaw, sDtype); err != nil {
		return nil, 0, 0, fmt.Errorf("per-expert scale decode %s: %w", sName, err)
	}
	if len(raw) < rows*cols {
		return nil, 0, 0, fmt.Errorf("per-expert %s raw bytes=%d want %d", wName, len(raw), rows*cols)
	}
	for r := 0; r < rows; r++ {
		scale := scales[0]
		if len(scales) != 1 {
			scale = scales[r]
		}
		base := r * cols
		for c := 0; c < cols; c++ {
			out[base+c] = diffusionGemmaFP8E4M3Table[raw[base+c]] * scale
		}
	}
	return out, rows, cols, nil
}

type decodedExpertWeights struct {
	gateW []float32
	upW   []float32
	downW []float32
}

type expertWeightLayout struct {
	nExperts     int
	intermediate int
	fused        bool
	layerPrefix  string
}

func expertLayoutForLayer(weights *TextWeights, lb TextLayerBindings, hiddenSize int) (expertWeightLayout, error) {
	if lb.ExpertsGateUpProj != nil && lb.ExpertsDownProj != nil && len(lb.ExpertsGateUpProj.Shape) == 3 && len(lb.ExpertsDownProj.Shape) == 3 {
		nExperts := lb.ExpertsGateUpProj.Shape[0]
		gateUpRows := lb.ExpertsGateUpProj.Shape[1]
		if gateUpRows%2 != 0 || lb.ExpertsGateUpProj.Shape[2] != hiddenSize || lb.ExpertsDownProj.Shape[0] != nExperts || lb.ExpertsDownProj.Shape[1] != hiddenSize || lb.ExpertsDownProj.Shape[2] != gateUpRows/2 {
			return expertWeightLayout{}, fmt.Errorf("DiffusionGemma expert fused shape mismatch gate_up=%v down=%v hidden=%d", lb.ExpertsGateUpProj.Shape, lb.ExpertsDownProj.Shape, hiddenSize)
		}
		return expertWeightLayout{nExperts: nExperts, intermediate: gateUpRows / 2, fused: true}, nil
	}
	if !lb.HasPerExpertWeights {
		return expertWeightLayout{}, fmt.Errorf("DiffusionGemma expert tensor bindings missing")
	}
	if lb.RouterProj == nil || len(lb.RouterProj.Shape) != 2 || lb.RouterProj.Shape[0] <= 0 {
		return expertWeightLayout{}, fmt.Errorf("DiffusionGemma per-expert layout missing router shape")
	}
	prefix := layerPrefixFromBinding(lb.RouterProj)
	if prefix == "" {
		prefix = layerPrefixFromBinding(lb.InputLayerNorm)
	}
	if prefix == "" {
		return expertWeightLayout{}, fmt.Errorf("DiffusionGemma per-expert layout could not infer layer prefix")
	}
	_, _, gateShape, err := weights.RawTensor(perExpertTensorName(prefix, 0, "gate_proj"))
	if err != nil {
		return expertWeightLayout{}, err
	}
	_, _, upShape, err := weights.RawTensor(perExpertTensorName(prefix, 0, "up_proj"))
	if err != nil {
		return expertWeightLayout{}, err
	}
	_, _, downShape, err := weights.RawTensor(perExpertTensorName(prefix, 0, "down_proj"))
	if err != nil {
		return expertWeightLayout{}, err
	}
	if len(gateShape) != 2 || len(upShape) != 2 || len(downShape) != 2 || gateShape[1] != hiddenSize || upShape[1] != hiddenSize || gateShape[0] != upShape[0] || downShape[0] != hiddenSize || downShape[1] != gateShape[0] {
		return expertWeightLayout{}, fmt.Errorf("DiffusionGemma per-expert shape mismatch gate=%v up=%v down=%v hidden=%d", gateShape, upShape, downShape, hiddenSize)
	}
	return expertWeightLayout{nExperts: lb.RouterProj.Shape[0], intermediate: gateShape[0], fused: false, layerPrefix: prefix}, nil
}

func loadLayerExpertWeights(weights *TextWeights, lb TextLayerBindings, layout expertWeightLayout, expertID, hiddenSize int) (decodedExpertWeights, error) {
	if expertID < 0 || expertID >= layout.nExperts {
		return decodedExpertWeights{}, fmt.Errorf("DiffusionGemma expert %d outside [0,%d)", expertID, layout.nExperts)
	}
	if layout.fused {
		guSlice, guRows, guCols, err := loadExpertSlice(weights, lb.ExpertsGateUpProj, expertID)
		if err != nil {
			return decodedExpertWeights{}, err
		}
		dSlice, downRows, downCols, err := loadExpertSlice(weights, lb.ExpertsDownProj, expertID)
		if err != nil {
			return decodedExpertWeights{}, err
		}
		if guRows != layout.intermediate*2 || guCols != hiddenSize || downRows != hiddenSize || downCols != layout.intermediate {
			return decodedExpertWeights{}, fmt.Errorf("DiffusionGemma expert %d fused slice mismatch gu=[%d,%d] down=[%d,%d]", expertID, guRows, guCols, downRows, downCols)
		}
		return decodedExpertWeights{gateW: guSlice[:layout.intermediate*hiddenSize], upW: guSlice[layout.intermediate*hiddenSize:], downW: dSlice}, nil
	}
	gateW, gateRows, gateCols, err := loadPerExpertFP8(weights, layout.layerPrefix, expertID, "gate_proj")
	if err != nil {
		return decodedExpertWeights{}, err
	}
	upW, upRows, upCols, err := loadPerExpertFP8(weights, layout.layerPrefix, expertID, "up_proj")
	if err != nil {
		return decodedExpertWeights{}, err
	}
	downW, downRows, downCols, err := loadPerExpertFP8(weights, layout.layerPrefix, expertID, "down_proj")
	if err != nil {
		return decodedExpertWeights{}, err
	}
	if gateRows != layout.intermediate || upRows != layout.intermediate || gateCols != hiddenSize || upCols != hiddenSize || downRows != hiddenSize || downCols != layout.intermediate {
		return decodedExpertWeights{}, fmt.Errorf("DiffusionGemma expert %d per-expert slice mismatch gate=[%d,%d] up=[%d,%d] down=[%d,%d]", expertID, gateRows, gateCols, upRows, upCols, downRows, downCols)
	}
	return decodedExpertWeights{gateW: gateW, upW: upW, downW: downW}, nil
}
