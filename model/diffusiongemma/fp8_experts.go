package diffusiongemma

import (
	"fmt"
	"math"

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

			// Load FP8 expert weights, dequant to F32 on CPU, run SIMD GEMV
			// (faster than 3 GPU round-trips per expert for small matrices)
			gateName := fmt.Sprintf("%s.%d.gate_proj", prefix, expertID)
			gateW, gateScale, gateShape, err := loadFP8Proj(fp8w.shards, gateName)
			if err != nil {
				return fmt.Errorf("FP8 expert %d gate: %w", expertID, err)
			}
			upName := fmt.Sprintf("%s.%d.up_proj", prefix, expertID)
			upW, upScale, upShape, err := loadFP8Proj(fp8w.shards, upName)
			if err != nil {
				return fmt.Errorf("FP8 expert %d up: %w", expertID, err)
			}
			downName := fmt.Sprintf("%s.%d.down_proj", prefix, expertID)
			downW, downScale, downShape, err := loadFP8Proj(fp8w.shards, downName)
			if err != nil {
				return fmt.Errorf("FP8 expert %d down: %w", expertID, err)
			}
			gateF32 := dequantFP8(gateW, gateScale, gateShape[0], gateShape[1])
			upF32 := dequantFP8(upW, upScale, upShape[0], upShape[1])
			downF32 := dequantFP8(downW, downScale, downShape[0], downShape[1])
			simd.GemvRows(gate, normedRow, gateF32, gateShape[0], gateShape[1])
			simd.GemvRows(up, normedRow, upF32, upShape[0], upShape[1])
			if !simd.GELUTanhMulTo(act, gate, up) {
				return fmt.Errorf("FP8 expert activation rejected")
			}
			simd.GemvRows(expertOut, act, downF32, downShape[0], downShape[1])
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

// dequantFP8 converts FP8 E4M3 bytes + per-channel scale to F32.
func dequantFP8(raw []byte, scale []float32, rows, cols int) []float32 {
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		s := float32(1.0)
		if r < len(scale) {
			s = scale[r]
		} else if len(scale) == 1 {
			s = scale[0]
		}
		for c := 0; c < cols; c++ {
			idx := r*cols + c
			if idx < len(raw) {
				out[idx] = fp8e4m3ToF32(raw[idx]) * s
			}
		}
	}
	return out
}

func fp8e4m3ToF32(b byte) float32 {
	sign := uint32(b&0x80) << 24
	exp := (b >> 3) & 0x0f
	mant := uint32(b & 0x07)
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// subnormal
		for mant&0x08 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &^= 0x08
	} else if exp == 15 {
		// NaN (no inf in E4M3)
		return math.Float32frombits(sign | 0x7f800000 | (mant << 20))
	}
	exp32 := uint32(int(exp) + (127 - 7))
	return math.Float32frombits(sign | (exp32 << 23) | (mant << 20))
}
