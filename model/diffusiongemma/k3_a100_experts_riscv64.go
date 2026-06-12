//go:build riscv64

package diffusiongemma

import (
	"fmt"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

type k3ExpertAssignment struct {
	pos    int
	weight float32
}

func k3RunPerExpertA100(weights *TextWeights, lb TextLayerBindings, layout expertWeightLayout, scratch ForwardScratch, preNorm2 []float32, hiddenSize, positions, topK int) (bool, error) {
	if !k3Enabled() || !k3A100Q8Enabled() || layout.fused || layout.layerPrefix == "" || positions <= 0 || topK <= 0 {
		return false, nil
	}
	normed := make([]float32, positions*hiddenSize)
	for pos := 0; pos < positions; pos++ {
		row := normed[pos*hiddenSize : (pos+1)*hiddenSize]
		copy(row, scratch.Residual[pos*hiddenSize:(pos+1)*hiddenSize])
		if !simd.RMSNormTo(row, preNorm2, 1e-6) {
			return true, fmt.Errorf("DiffusionGemma K3 A100 expert pre_norm_2 rejected")
		}
	}
	assignments := map[int][]k3ExpertAssignment{}
	for pos := 0; pos < positions; pos++ {
		for k := 0; k < topK; k++ {
			expertID := scratch.TopKIDs[pos*topK+k]
			if expertID < 0 || expertID >= layout.nExperts {
				continue
			}
			assignments[expertID] = append(assignments[expertID], k3ExpertAssignment{pos: pos, weight: scratch.TopKVals[pos*topK+k]})
		}
	}
	for expertID := range assignments {
		for _, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
			name := perExpertTensorName(layout.layerPrefix, expertID, proj)
			if _, ok, err := k3Q80ForTensorName(weights, name); err != nil || !ok {
				return ok, err
			}
		}
	}
	for expertID, rows := range assignments {
		batch := len(rows)
		if batch == 0 {
			continue
		}
		x := make([]float32, batch*hiddenSize)
		for i, a := range rows {
			copy(x[i*hiddenSize:(i+1)*hiddenSize], normed[a.pos*hiddenSize:(a.pos+1)*hiddenSize])
		}
		gate := make([]float32, batch*layout.intermediate)
		up := make([]float32, batch*layout.intermediate)
		gateName := perExpertTensorName(layout.layerPrefix, expertID, "gate_proj")
		upName := perExpertTensorName(layout.layerPrefix, expertID, "up_proj")
		done, err := k3Gemm2RowsQ80Names(gate, up, x, batch, weights, gateName, upName)
		if err != nil || !done {
			return done, err
		}
		act := make([]float32, batch*layout.intermediate)
		for i := 0; i < batch; i++ {
			if !simd.GELUTanhMulTo(act[i*layout.intermediate:(i+1)*layout.intermediate], gate[i*layout.intermediate:(i+1)*layout.intermediate], up[i*layout.intermediate:(i+1)*layout.intermediate]) {
				return true, fmt.Errorf("DiffusionGemma K3 A100 expert activation rejected")
			}
		}
		down := make([]float32, batch*hiddenSize)
		downName := perExpertTensorName(layout.layerPrefix, expertID, "down_proj")
		done, err = k3GemmRowsQ80Name(down, act, batch, weights, downName)
		if err != nil || !done {
			return done, err
		}
		for i, a := range rows {
			dst := scratch.MoeOut[a.pos*hiddenSize : (a.pos+1)*hiddenSize]
			src := down[i*hiddenSize : (i+1)*hiddenSize]
			k3SaxpyV(a.weight, src, dst)
		}
	}
	return true, nil
}
