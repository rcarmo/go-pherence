package diffusiongemma

import (
	"fmt"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func runGGUFCPUExpertsGrouped(op LayerOp, weights *TextWeights, scratch ForwardScratch, idx *GGUFExpertIndex, arrays SelectedExpertGroupedArrays) error {
	if idx == nil || weights == nil {
		return fmt.Errorf("GGUF grouped CPU experts missing weights/index")
	}
	if op.Layer < 0 || op.Layer >= idx.NumLayers {
		return fmt.Errorf("GGUF grouped CPU experts layer %d outside index", op.Layer)
	}
	if err := arrays.Validate(); err != nil {
		return err
	}
	hidden, intermediate := idx.HiddenSize, idx.Intermediate
	if hidden <= 0 || intermediate <= 0 || len(scratch.Residual)%hidden != 0 || len(scratch.MoeOut) < len(scratch.Residual) {
		return fmt.Errorf("GGUF grouped CPU experts invalid hidden/residual hidden=%d residual=%d", hidden, len(scratch.Residual))
	}
	fp := weights.ForwardPlan()
	if op.Layer >= len(fp.Layers) {
		return fmt.Errorf("GGUF grouped CPU experts layer %d outside plan", op.Layer)
	}
	preNorm2, err := loadFloatVector(weights, fp.Layers[op.Layer].PreFFNLayerNorm2)
	if err != nil {
		return err
	}
	if len(preNorm2) != hidden {
		return fmt.Errorf("GGUF grouped CPU experts preNorm hidden=%d want %d", len(preNorm2), hidden)
	}
	for i := range scratch.MoeOut {
		scratch.MoeOut[i] = 0
	}
	for g, expert := range arrays.ActiveExperts {
		if expert < 0 || expert >= idx.NumExperts {
			return fmt.Errorf("GGUF grouped CPU expert %d outside count %d", expert, idx.NumExperts)
		}
		for wi := arrays.Offsets[g]; wi < arrays.Offsets[g+1]; wi++ {
			pos := arrays.WorkPositions[wi]
			if pos < 0 || (pos+1)*hidden > len(scratch.Residual) {
				return fmt.Errorf("GGUF grouped CPU work pos=%d outside residual", pos)
			}
			normed := append([]float32(nil), scratch.Residual[pos*hidden:(pos+1)*hidden]...)
			if !simd.RMSNormTo(normed, preNorm2, 1e-6) {
				return fmt.Errorf("GGUF grouped CPU pre_norm_2 rejected")
			}
			expertOut := make([]float32, hidden)
			if err := idx.RunGGUFExpertMLP(op.Layer, expert, normed, expertOut); err != nil {
				return err
			}
			// RunGGUFExpertMLP already applies per-expert down scale, matching
			// runGGUFCPUExpertsIndexed. The fused GPU kernel consumes
			// WorkDownScales separately because it operates below that helper.
			w := arrays.WorkWeights[wi]
			for i, v := range expertOut {
				scratch.MoeOut[pos*hidden+i] += w * v
			}
		}
	}
	postNorm2, err := loadFloatVector(weights, fp.Layers[op.Layer].PostFFNLayerNorm2)
	if err != nil {
		return err
	}
	for off := 0; off < len(scratch.MoeOut); off += hidden {
		if !simd.RMSNormTo(scratch.MoeOut[off:off+hidden], postNorm2, 1e-6) {
			return fmt.Errorf("GGUF grouped CPU post_norm_2 rejected")
		}
	}
	return nil
}
