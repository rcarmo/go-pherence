package diffusiongemma

import (
	"fmt"
	"os"
	"time"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// GPUFP8ExpertPool holds GPU-resident FP8 expert weights for multiple layers.
type GPUFP8ExpertPool struct {
	// Layers maps layer index → expert index → GPU FP8 linears
	Layers map[int]map[int]*GPUFP8ExpertLinears
}

// GPUFP8ExpertLinears holds the three projections for one expert on GPU.
type GPUFP8ExpertLinears struct {
	Gate *gpu.GPUFP8E4M3Linear
	Up   *gpu.GPUFP8E4M3Linear
	Down *gpu.GPUFP8E4M3Linear
}

// UploadExpertPool uploads all 128 experts for the specified layers to GPU.
func UploadExpertPool(fp8w *FP8TextWeights, layers []int, progress bool) (*GPUFP8ExpertPool, error) {
	pool := &GPUFP8ExpertPool{Layers: make(map[int]map[int]*GPUFP8ExpertLinears)}
	for _, layer := range layers {
		started := time.Now()
		experts := make(map[int]*GPUFP8ExpertLinears, 128)
		prefix := fmt.Sprintf("model.decoder.layers.%d.experts", layer)
		for eid := 0; eid < 128; eid++ {
			el, err := uploadOneExpert(fp8w, prefix, eid)
			if err != nil {
				return nil, fmt.Errorf("layer %d expert %d: %w", layer, eid, err)
			}
			experts[eid] = el
		}
		pool.Layers[layer] = experts
		if progress {
			fmt.Fprintf(os.Stderr, "DiffusionGemma FP8: uploaded 128 experts for layer %d elapsed=%s\n", layer, time.Since(started).Round(time.Millisecond))
		}
	}
	return pool, nil
}

func uploadOneExpert(fp8w *FP8TextWeights, prefix string, eid int) (*GPUFP8ExpertLinears, error) {
	gW, gS, gSh, err := loadFP8Proj(fp8w.shards, fmt.Sprintf("%s.%d.gate_proj", prefix, eid))
	if err != nil {
		return nil, err
	}
	uW, uS, uSh, err := loadFP8Proj(fp8w.shards, fmt.Sprintf("%s.%d.up_proj", prefix, eid))
	if err != nil {
		return nil, err
	}
	dW, dS, dSh, err := loadFP8Proj(fp8w.shards, fmt.Sprintf("%s.%d.down_proj", prefix, eid))
	if err != nil {
		return nil, err
	}
	gate, err := gpu.UploadFP8E4M3Linear(gW, gS, nil, gSh[0], gSh[1])
	if err != nil {
		return nil, err
	}
	up, err := gpu.UploadFP8E4M3Linear(uW, uS, nil, uSh[0], uSh[1])
	if err != nil {
		return nil, err
	}
	down, err := gpu.UploadFP8E4M3Linear(dW, dS, nil, dSh[0], dSh[1])
	if err != nil {
		return nil, err
	}
	return &GPUFP8ExpertLinears{Gate: gate, Up: up, Down: down}, nil
}

// runGPUResidentExperts runs MoE using pre-uploaded GPU-resident expert weights.
func runGPUResidentExperts(op LayerOp, weights *TextWeights, scratch ForwardScratch, pool *GPUFP8ExpertPool) error {
	experts, ok := pool.Layers[op.Layer]
	if !ok {
		return fmt.Errorf("no GPU-resident experts for layer %d", op.Layer)
	}
	fp := weights.ForwardPlan()
	if op.Layer < 0 || op.Layer >= len(fp.Layers) {
		return fmt.Errorf("layer %d outside plan", op.Layer)
	}
	lb := fp.Layers[op.Layer]
	preNorm2, err := loadFloatVector(weights, lb.PreFFNLayerNorm2)
	if err != nil {
		return err
	}
	hiddenSize := len(preNorm2)
	positions := len(scratch.Residual) / hiddenSize
	topK := len(scratch.TopKIDs) / positions

	for i := range scratch.MoeOut {
		scratch.MoeOut[i] = 0
	}

	intermediate := 704
	normedRow := make([]float32, hiddenSize)
	gate := make([]float32, intermediate)
	up := make([]float32, intermediate)
	act := make([]float32, intermediate)
	expertOut := make([]float32, hiddenSize)

	for pos := 0; pos < positions; pos++ {
		resRow := scratch.Residual[pos*hiddenSize : (pos+1)*hiddenSize]
		copy(normedRow, resRow)
		if !simd.RMSNormTo(normedRow, preNorm2, 1e-6) {
			return fmt.Errorf("pre_norm_2 rejected")
		}
		dst := scratch.MoeOut[pos*hiddenSize : (pos+1)*hiddenSize]
		for k := 0; k < topK; k++ {
			eid := scratch.TopKIDs[pos*topK+k]
			w := scratch.TopKVals[pos*topK+k]
			el, ok := experts[eid]
			if !ok || el == nil {
				continue
			}
			gpu.GemvFP8E4M3(gate, normedRow, el.Gate)
			gpu.GemvFP8E4M3(up, normedRow, el.Up)
			simd.GELUTanhMulTo(act, gate, up)
			gpu.GemvFP8E4M3(expertOut, act, el.Down)
			for i := range dst {
				dst[i] += w * expertOut[i]
			}
		}
	}

	postNorm2, err := loadFloatVector(weights, lb.PostFFNLayerNorm2)
	if err != nil {
		return err
	}
	for off := 0; off < len(scratch.MoeOut); off += hiddenSize {
		simd.RMSNormTo(scratch.MoeOut[off:off+hiddenSize], postNorm2, 1e-6)
	}
	return nil
}
