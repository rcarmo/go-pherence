package diffusiongemma

import (
	"fmt"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// runFP8ExpertsFromResidual runs MoE experts using FP8 weights from the FP8 checkpoint,
// loading only the selected top-k experts per position via GPU FP8 GEMV.
// Uses reusable GPU buffers to avoid OOM from accumulated allocations.
func runFP8ExpertsFromResidual(op LayerOp, weights *TextWeights, scratch ForwardScratch, fp8w *FP8TextWeights) error {
	if weights == nil || fp8w == nil {
		return fmt.Errorf("DiffusionGemma FP8 experts missing weights")
	}
	fp := weights.ForwardPlan()
	if op.Layer < 0 || op.Layer >= len(fp.Layers) {
		return fmt.Errorf("DiffusionGemma FP8 experts layer %d outside plan", op.Layer)
	}
	lb := fp.Layers[op.Layer]
	preNorm2, err := loadFloatVector(weights, lb.PreFFNLayerNorm2)
	if err != nil {
		return err
	}
	hiddenSize := len(preNorm2)
	if hiddenSize <= 0 || len(scratch.Residual)%hiddenSize != 0 {
		return fmt.Errorf("DiffusionGemma FP8 expert hidden mismatch")
	}
	positions := len(scratch.Residual) / hiddenSize
	topK := len(scratch.TopKIDs) / positions

	prefix := fmt.Sprintf("model.decoder.layers.%d.experts", op.Layer)

	// Reusable GPU buffers for one expert at a time
	var gateGPU, upGPU, downGPU *gpu.GPUFP8E4M3Linear

	intermediate := 704 // from config

	for i := range scratch.MoeOut {
		scratch.MoeOut[i] = 0
	}
	normedRow := make([]float32, hiddenSize)
	gate := make([]float32, intermediate)
	up := make([]float32, intermediate)
	act := make([]float32, intermediate)
	expertOut := make([]float32, hiddenSize)

	for pos := 0; pos < positions; pos++ {
		resRow := scratch.Residual[pos*hiddenSize : (pos+1)*hiddenSize]
		copy(normedRow, resRow)
		if !simd.RMSNormTo(normedRow, preNorm2, 1e-6) {
			return fmt.Errorf("DiffusionGemma FP8 expert pre_norm_2 rejected")
		}
		dst := scratch.MoeOut[pos*hiddenSize : (pos+1)*hiddenSize]
		for k := 0; k < topK; k++ {
			expertID := scratch.TopKIDs[pos*topK+k]
			weight := scratch.TopKVals[pos*topK+k]
			if expertID < 0 {
				continue
			}

			// Load FP8 expert weights and upload/reuse GPU buffers
			gateName := fmt.Sprintf("%s.%d.gate_proj", prefix, expertID)
			gateW, gateScale, gateShape, err := loadFP8Proj(fp8w.shards, gateName)
			if err != nil {
				return fmt.Errorf("FP8 expert %d gate: %w", expertID, err)
			}
			if err := gpu.UploadFP8E4M3LinearReuse(&gateGPU, gateW, gateScale, nil, gateShape[0], gateShape[1]); err != nil {
				return fmt.Errorf("FP8 expert %d gate upload: %w", expertID, err)
			}

			upName := fmt.Sprintf("%s.%d.up_proj", prefix, expertID)
			upW, upScale, upShape, err := loadFP8Proj(fp8w.shards, upName)
			if err != nil {
				return fmt.Errorf("FP8 expert %d up: %w", expertID, err)
			}
			if err := gpu.UploadFP8E4M3LinearReuse(&upGPU, upW, upScale, nil, upShape[0], upShape[1]); err != nil {
				return fmt.Errorf("FP8 expert %d up upload: %w", expertID, err)
			}

			downName := fmt.Sprintf("%s.%d.down_proj", prefix, expertID)
			downW, downScale, downShape, err := loadFP8Proj(fp8w.shards, downName)
			if err != nil {
				return fmt.Errorf("FP8 expert %d down: %w", expertID, err)
			}
			if err := gpu.UploadFP8E4M3LinearReuse(&downGPU, downW, downScale, nil, downShape[0], downShape[1]); err != nil {
				return fmt.Errorf("FP8 expert %d down upload: %w", expertID, err)
			}

			if err := gpu.GemvFP8E4M3(gate, normedRow, gateGPU); err != nil {
				return fmt.Errorf("FP8 expert %d gate GEMV: %w", expertID, err)
			}
			if err := gpu.GemvFP8E4M3(up, normedRow, upGPU); err != nil {
				return fmt.Errorf("FP8 expert %d up GEMV: %w", expertID, err)
			}
			if !simd.GELUTanhMulTo(act, gate, up) {
				return fmt.Errorf("FP8 expert activation rejected")
			}
			if err := gpu.GemvFP8E4M3(expertOut, act, downGPU); err != nil {
				return fmt.Errorf("FP8 expert %d down GEMV: %w", expertID, err)
			}
			for i := range dst {
				dst[i] += weight * expertOut[i]
			}
		}
	}

	postNorm2, err := loadFloatVector(weights, lb.PostFFNLayerNorm2)
	if err != nil {
		return err
	}
	for off := 0; off < len(scratch.MoeOut); off += hiddenSize {
		if !simd.RMSNormTo(scratch.MoeOut[off:off+hiddenSize], postNorm2, 1e-6) {
			return fmt.Errorf("DiffusionGemma FP8 expert post_norm_2 rejected")
		}
	}
	return nil
}
