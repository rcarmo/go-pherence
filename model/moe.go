package model

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"

	"github.com/rcarmo/go-pherence/backends/mlx"
	"github.com/rcarmo/go-pherence/half"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// LoadSwitchMLXExperts loads a switch_mlp-style 3D packed tensor and
// slices it into per-expert mlx.QuantWeight entries.
//
// The safetensors weight has shape [numExperts, outDim, packedInDim] (U32)
// with matching scales/biases [numExperts, outDim, numGroups] (BF16/F32).
func LoadSwitchMLXExperts(
	f interface {
		GetRaw(name string) ([]byte, string, []int, error)
	},
	baseName string,
	numExperts, outDim, inDim, groupSize, bits int,
) ([]*mlx.QuantWeight, error) {
	if f == nil {
		return nil, fmt.Errorf("nil safetensors source")
	}
	if numExperts <= 0 || outDim <= 0 || inDim <= 0 || groupSize <= 0 || bits <= 0 || bits > 32 || 32%bits != 0 {
		return nil, fmt.Errorf("invalid switch MLX dims experts=%d out=%d in=%d groupSize=%d bits=%d", numExperts, outDim, inDim, groupSize, bits)
	}
	if inDim%groupSize != 0 {
		return nil, fmt.Errorf("switch MLX inDim=%d is not divisible by groupSize=%d", inDim, groupSize)
	}
	packFactor := 32 / bits
	if inDim%packFactor != 0 {
		return nil, fmt.Errorf("switch MLX inDim=%d is not divisible by packFactor=%d", inDim, packFactor)
	}
	// Load packed weight
	wRaw, wDtype, wShape, err := f.GetRaw(baseName + ".weight")
	if err != nil {
		return nil, fmt.Errorf("load %s.weight: %w", baseName, err)
	}
	if len(wShape) != 3 || wShape[0] != numExperts {
		return nil, fmt.Errorf("%s.weight: expected [%d, ?, ?], got %v", baseName, numExperts, wShape)
	}
	if wDtype != "U32" && wDtype != "I32" {
		return nil, fmt.Errorf("%s.weight: unsupported dtype %s (expected U32/I32)", baseName, wDtype)
	}

	// Load scales
	sRaw, sDtype, sShape, err := f.GetRaw(baseName + ".scales")
	if err != nil {
		return nil, fmt.Errorf("load %s.scales: %w", baseName, err)
	}
	if len(sShape) != 3 || sShape[0] != numExperts {
		return nil, fmt.Errorf("%s.scales: expected [%d, ?, ?], got %v", baseName, numExperts, sShape)
	}

	// Load biases
	bRaw, bDtype, bShape, err := f.GetRaw(baseName + ".biases")
	if err != nil {
		return nil, fmt.Errorf("load %s.biases: %w", baseName, err)
	}
	if len(bShape) != 3 || bShape[0] != numExperts {
		return nil, fmt.Errorf("%s.biases: expected [%d, ?, ?], got %v", baseName, numExperts, bShape)
	}

	numGroups := inDim / groupSize
	packedPerRow := inDim / packFactor

	// Verify shapes
	if wShape[1] != outDim || wShape[2] != packedPerRow {
		return nil, fmt.Errorf("%s.weight: expected [%d, %d, %d], got %v",
			baseName, numExperts, outDim, packedPerRow, wShape)
	}

	// Per-expert slicing
	wElems, ok := checkedProduct(outDim, packedPerRow)
	if !ok {
		return nil, fmt.Errorf("%s.weight per-expert element count overflows", baseName)
	}
	sbElems, ok := checkedProduct(outDim, numGroups)
	if !ok {
		return nil, fmt.Errorf("%s scale/bias per-expert element count overflows", baseName)
	}
	wStride, ok := checkedProduct(wElems, 4) // bytes per expert in weight
	if !ok {
		return nil, fmt.Errorf("%s.weight byte stride overflows", baseName)
	}
	sElemBytes, err := switchMLXFloatElemBytes(sDtype)
	if err != nil {
		return nil, fmt.Errorf("%s.scales: %w", baseName, err)
	}
	bElemBytes, err := switchMLXFloatElemBytes(bDtype)
	if err != nil {
		return nil, fmt.Errorf("%s.biases: %w", baseName, err)
	}
	sStride, ok := checkedProduct(sbElems, sElemBytes)
	if !ok {
		return nil, fmt.Errorf("%s.scales byte stride overflows", baseName)
	}
	bStride, ok := checkedProduct(sbElems, bElemBytes)
	if !ok {
		return nil, fmt.Errorf("%s.biases byte stride overflows", baseName)
	}
	wantW, ok := checkedProduct(wStride, numExperts)
	if !ok {
		return nil, fmt.Errorf("%s.weight total byte size overflows", baseName)
	}
	wantS, ok := checkedProduct(sStride, numExperts)
	if !ok {
		return nil, fmt.Errorf("%s.scales total byte size overflows", baseName)
	}
	wantB, ok := checkedProduct(bStride, numExperts)
	if !ok {
		return nil, fmt.Errorf("%s.biases total byte size overflows", baseName)
	}
	if len(wRaw) < wantW || len(sRaw) < wantS || len(bRaw) < wantB {
		return nil, fmt.Errorf("%s: raw tensor data shorter than expected expert strides", baseName)
	}

	experts := make([]*mlx.QuantWeight, numExperts)
	for e := 0; e < numExperts; e++ {
		wSlice := wRaw[e*wStride : (e+1)*wStride]
		sSlice := sRaw[e*sStride : (e+1)*sStride]
		bSlice := bRaw[e*bStride : (e+1)*bStride]

		// Parse uint32 weight
		nW := len(wSlice) / 4
		weight := make([]uint32, nW)
		for i := 0; i < nW; i++ {
			weight[i] = binary.LittleEndian.Uint32(wSlice[i*4:])
		}

		scales, err := decodeSwitchMLXFloat(sSlice, sDtype)
		if err != nil {
			return nil, fmt.Errorf("%s.scales expert %d: %w", baseName, e, err)
		}
		biases, err := decodeSwitchMLXFloat(bSlice, bDtype)
		if err != nil {
			return nil, fmt.Errorf("%s.biases expert %d: %w", baseName, e, err)
		}

		experts[e] = &mlx.QuantWeight{
			Weight:    weight,
			Scales:    scales,
			Biases:    biases,
			InDim:     inDim,
			OutDim:    outDim,
			Groups:    numGroups,
			GroupSize: groupSize,
			Bits:      bits,
		}
	}

	return experts, nil
}

func switchMLXFloatElemBytes(dtype string) (int, error) {
	switch dtype {
	case "BF16", "F16":
		return 2, nil
	case "F32":
		return 4, nil
	default:
		return 0, fmt.Errorf("unsupported dtype %s (expected BF16/F16/F32)", dtype)
	}
}

func decodeSwitchMLXFloat(raw []byte, dtype string) ([]float32, error) {
	elemBytes, err := switchMLXFloatElemBytes(dtype)
	if err != nil {
		return nil, err
	}
	if len(raw)%elemBytes != 0 {
		return nil, fmt.Errorf("raw byte length %d is not divisible by dtype size %d", len(raw), elemBytes)
	}
	out := make([]float32, len(raw)/elemBytes)
	switch dtype {
	case "BF16":
		for i := range out {
			out[i] = half.BF16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case "F16":
		for i := range out {
			out[i] = half.F16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case "F32":
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	}
	return out, nil
}

// moeForward runs the MoE forward pass: router → top-k → expert MLPs → weighted sum.
func moeForward(x []float32, layer *LlamaLayer, cfg LlamaConfig) []float32 {
	return moeForwardWithREAP(x, layer, cfg, nil, -1)
}

func moeForwardWithREAP(x []float32, layer *LlamaLayer, cfg LlamaConfig, reap *REAPConfig, layerIdx int) []float32 {
	if layer == nil || len(x) == 0 || cfg.NumExperts <= 0 || cfg.MoEIntermediate <= 0 {
		return nil
	}
	h := len(x)
	numExperts := cfg.NumExperts
	numActive := cfg.NumExpertsPerTok
	if numActive <= 0 {
		numActive = 8
	}
	if numActive > numExperts {
		numActive = numExperts
	}

	// Router: compute logits for each expert
	routerLogits := make([]float32, numExperts)
	if layer.RouterW != nil {
		if !mlx.GemvTo(routerLogits, x, layer.RouterW) {
			return nil
		}
	}

	// Softmax over router logits.
	if !simd.SoftmaxInPlace(routerLogits) {
		return nil
	}

	// Top-k selection. REAP masks statically pruned experts before choosing the
	// active route set, preserving pure Go/SIMD execution while matching pruned
	// MoE checkpoints that omit or intentionally disable cold experts.
	type expertScore struct {
		id    int
		score float32
	}
	selected := make([]expertScore, 0, numActive)
	for i := 0; i < numActive; i++ {
		bestID := -1
		bestScore := float32(-1)
		for j, s := range routerLogits {
			if !reap.Allows(layerIdx, j) {
				continue
			}
			if s > bestScore {
				// Check not already selected
				alreadyPicked := false
				for _, sel := range selected {
					if sel.id == j {
						alreadyPicked = true
						break
					}
				}
				if !alreadyPicked {
					bestID = j
					bestScore = s
				}
			}
		}
		if bestID >= 0 {
			selected = append(selected, expertScore{id: bestID, score: bestScore})
		}
	}

	// Normalize selected weights (norm_topk_prob)
	if len(selected) == 0 {
		return make([]float32, h)
	}

	if cfg.NormTopKProb {
		var sum float32
		for _, s := range selected {
			sum += s.score
		}
		if sum > 0 {
			for i := range selected {
				selected[i].score /= sum
			}
		}
	}

	// Run selected experts in parallel and accumulate weighted output.
	// Each expert gets a deterministic slot inside one contiguous scratch buffer,
	// so gate/up/down temporaries are reused without changing expert completion or
	// weighted accumulation order.
	moeInter := cfg.MoEIntermediate
	out := make([]float32, h)
	slotElems, ok := checkedAddNonNegative(moeInter, moeInter)
	if !ok {
		return nil
	}
	slotElems, ok = checkedAddNonNegative(slotElems, h)
	if !ok {
		return nil
	}
	totalScratch, ok := checkedProduct(slotElems, len(selected))
	if !ok {
		return nil
	}
	expertScratch := make([]float32, totalScratch)

	type expertResult struct {
		weight float32
		ok     bool
	}
	results := make([]expertResult, len(selected))
	var wg sync.WaitGroup
	for si, exp := range selected {
		eid := exp.id
		if eid < 0 || eid >= len(layer.ExpertGateW) || eid >= len(layer.ExpertUpW) || eid >= len(layer.ExpertDownW) || layer.ExpertGateW[eid] == nil || layer.ExpertUpW[eid] == nil || layer.ExpertDownW[eid] == nil {
			continue
		}
		slotBase := si * slotElems
		gate := expertScratch[slotBase : slotBase+moeInter]
		up := expertScratch[slotBase+moeInter : slotBase+2*moeInter]
		down := expertScratch[slotBase+2*moeInter : slotBase+slotElems]
		wg.Add(1)
		go func(idx int, expertID int, w float32, gate, up, down []float32) {
			defer wg.Done()
			// Expert MLP: gate_proj → SiLU × up_proj → down_proj
			if !mlx.Gemv2To(gate, up, x, layer.ExpertGateW[expertID], layer.ExpertUpW[expertID]) {
				return
			}
			if !simd.SiLUMulTo(gate, gate, up) {
				return
			}
			if !mlx.GemvTo(down, gate, layer.ExpertDownW[expertID]) {
				return
			}
			results[idx] = expertResult{weight: w, ok: true}
		}(si, eid, exp.score, gate, up, down)
	}
	wg.Wait()

	for idx, r := range results {
		if !r.ok {
			continue
		}
		slotBase := idx * slotElems
		down := expertScratch[slotBase+2*moeInter : slotBase+slotElems]
		for i := range out {
			out[i] += r.weight * down[i]
		}
	}

	return out
}
