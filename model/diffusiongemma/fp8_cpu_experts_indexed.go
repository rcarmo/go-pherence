package diffusiongemma

import (
	"fmt"
	"log"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/rcarmo/go-pherence/backends/simd/quant/fp8"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
)

// FP8ExpertIndex is a pre-built direct index from (layer, expert, projection)
// to fp8.Linear wrappers. Eliminates per-call string formatting and
// safetensors index lookup. Uses AVX2 SIMD GEMV via fp8.Linear.GemvTo.
type FP8ExpertIndex struct {
	NumLayers    int
	NumExperts   int
	HiddenSize   int
	Intermediate int
	// entries[layer][expert] holds the three fp8.Linear projections
	entries [][]fp8ExpertEntry
}

type fp8ExpertEntry struct {
	gate, up, down fp8.Linear
}

// BuildFP8ExpertIndex pre-resolves all expert weight pointers from the FP8 checkpoint.
// This is done once at startup; subsequent lookups are O(1) array indexing.
func BuildFP8ExpertIndex(fp8w *FP8TextWeights, numLayers, numExperts int) (*FP8ExpertIndex, error) {
	if fp8w == nil || fp8w.shards == nil {
		return nil, fmt.Errorf("FP8 expert index missing weights")
	}
	if numLayers <= 0 || numExperts <= 0 || numLayers > len(fp8w.Layers) {
		return nil, fmt.Errorf("FP8 expert index invalid dimensions layers=%d/%d experts=%d", numLayers, len(fp8w.Layers), numExperts)
	}
	t0 := time.Now()
	idx := &FP8ExpertIndex{
		NumLayers:  numLayers,
		NumExperts: numExperts,
		entries:    make([][]fp8ExpertEntry, numLayers),
	}
	for l := 0; l < numLayers; l++ {
		idx.entries[l] = make([]fp8ExpertEntry, numExperts)
		prefix := fmt.Sprintf("model.decoder.layers.%d.experts", l)
		for e := 0; e < numExperts; e++ {
			ep := fmt.Sprintf("%s.%d", prefix, e)
			gW, gS, gSh, err := loadFP8Proj(fp8w.shards, ep+".gate_proj")
			if err != nil {
				return nil, fmt.Errorf("index layer %d expert %d gate: %w", l, e, err)
			}
			uW, uS, uSh, err := loadFP8Proj(fp8w.shards, ep+".up_proj")
			if err != nil {
				return nil, fmt.Errorf("index layer %d expert %d up: %w", l, e, err)
			}
			dW, dS, dSh, err := loadFP8Proj(fp8w.shards, ep+".down_proj")
			if err != nil {
				return nil, fmt.Errorf("index layer %d expert %d down: %w", l, e, err)
			}
			if gSh[0] <= 0 || gSh[1] <= 0 || uSh[0] != gSh[0] || uSh[1] != gSh[1] || dSh[0] != gSh[1] || dSh[1] != gSh[0] {
				return nil, fmt.Errorf("index layer %d expert %d shape mismatch gate=[%d,%d] up=[%d,%d] down=[%d,%d]", l, e, gSh[0], gSh[1], uSh[0], uSh[1], dSh[0], dSh[1])
			}
			if idx.HiddenSize == 0 {
				idx.HiddenSize = gSh[1]
				idx.Intermediate = gSh[0]
			} else if idx.HiddenSize != gSh[1] || idx.Intermediate != gSh[0] {
				return nil, fmt.Errorf("index layer %d expert %d dims hidden=%d/%d intermediate=%d/%d", l, e, gSh[1], idx.HiddenSize, gSh[0], idx.Intermediate)
			}
			entry := fp8ExpertEntry{
				gate: fp8.Linear{OutDim: gSh[0], InDim: gSh[1], Weight: gW, Scale: gS},
				up:   fp8.Linear{OutDim: uSh[0], InDim: uSh[1], Weight: uW, Scale: uS},
				down: fp8.Linear{OutDim: dSh[0], InDim: dSh[1], Weight: dW, Scale: dS},
			}
			if err := entry.gate.Validate(); err != nil {
				return nil, fmt.Errorf("index layer %d expert %d gate: %w", l, e, err)
			}
			if err := entry.up.Validate(); err != nil {
				return nil, fmt.Errorf("index layer %d expert %d up: %w", l, e, err)
			}
			if err := entry.down.Validate(); err != nil {
				return nil, fmt.Errorf("index layer %d expert %d down: %w", l, e, err)
			}
			idx.entries[l][e] = entry
		}
	}
	log.Printf("FP8 expert index: %d layers × %d experts built in %s", numLayers, numExperts, time.Since(t0).Round(time.Millisecond))
	return idx, nil
}

// expertWorkerScratch holds pre-allocated buffers for one expert worker.
// Reused across experts within the same worker to avoid GC pressure.
type expertWorkerScratch struct {
	batchIn   []float32 // [maxBatch * hiddenSize]
	batchGate []float32 // [maxBatch * intermediate]
	batchUp   []float32 // [maxBatch * intermediate]
	batchAct  []float32 // [maxBatch * intermediate]
	batchDown []float32 // [maxBatch * hiddenSize]
	xq        []float32 // dynamic-token quantized activation scratch, max(maxBatch*hiddenSize, maxBatch*intermediate)
	wf32      []float32 // [max(hiddenSize, intermediate)] for FP8 dequant
}

func (s *expertWorkerScratch) ensure(maxBatch, hiddenSize, intermediate int) error {
	inNeed, okIn := checked.MulInt(maxBatch, hiddenSize)
	midNeed, okMid := checked.MulInt(maxBatch, intermediate)
	if maxBatch <= 0 || hiddenSize <= 0 || intermediate <= 0 || !okIn || !okMid {
		return fmt.Errorf("FP8 expert scratch size overflow batch=%d hidden=%d intermediate=%d", maxBatch, hiddenSize, intermediate)
	}
	wfNeed := hiddenSize
	if intermediate > wfNeed {
		wfNeed = intermediate
	}
	if cap(s.batchIn) < inNeed {
		s.batchIn = make([]float32, inNeed)
	} else {
		s.batchIn = s.batchIn[:inNeed]
	}
	if cap(s.batchGate) < midNeed {
		s.batchGate = make([]float32, midNeed)
	} else {
		s.batchGate = s.batchGate[:midNeed]
	}
	if cap(s.batchUp) < midNeed {
		s.batchUp = make([]float32, midNeed)
	} else {
		s.batchUp = s.batchUp[:midNeed]
	}
	if cap(s.batchAct) < midNeed {
		s.batchAct = make([]float32, midNeed)
	} else {
		s.batchAct = s.batchAct[:midNeed]
	}
	if cap(s.batchDown) < inNeed {
		s.batchDown = make([]float32, inNeed)
	} else {
		s.batchDown = s.batchDown[:inNeed]
	}
	xqNeed := inNeed
	if midNeed > xqNeed {
		xqNeed = midNeed
	}
	if cap(s.xq) < xqNeed {
		s.xq = make([]float32, xqNeed)
	} else {
		s.xq = s.xq[:xqNeed]
	}
	if cap(s.wf32) < wfNeed {
		s.wf32 = make([]float32, wfNeed)
	} else {
		s.wf32 = s.wf32[:wfNeed]
	}
	return nil
}

// runFP8CPUExpertsIndexed runs MoE experts using the pre-built FP8 index.
// Uses dequant-once F32 SIMD dot and processes experts in parallel across CPU cores.
// Batches all positions for each expert to amortize weight reads.
func runFP8CPUExpertsIndexed(op LayerOp, weights *TextWeights, scratch ForwardScratch, idx *FP8ExpertIndex) error {
	if weights == nil || idx == nil {
		return fmt.Errorf("FP8 CPU experts: missing weights or index")
	}
	fp := weights.ForwardPlan()
	if op.Layer < 0 || op.Layer >= len(fp.Layers) || op.Layer >= idx.NumLayers {
		return fmt.Errorf("layer %d outside plan/index", op.Layer)
	}
	lb := fp.Layers[op.Layer]
	preNorm2, err := loadFloatVector(weights, lb.PreFFNLayerNorm2)
	if err != nil {
		return err
	}
	hiddenSize := len(preNorm2)
	if hiddenSize <= 0 || len(scratch.Residual)%hiddenSize != 0 || len(scratch.MoeOut) < len(scratch.Residual) {
		return fmt.Errorf("FP8 CPU experts: invalid hidden/residual size hidden=%d residual=%d moe=%d", hiddenSize, len(scratch.Residual), len(scratch.MoeOut))
	}
	positions := len(scratch.Residual) / hiddenSize
	if positions <= 0 {
		return fmt.Errorf("FP8 CPU experts: no positions")
	}
	topK := scratch.TopKExperts
	if topK <= 0 {
		topK = len(scratch.TopKIDs) / positions
	}
	if topK <= 0 || len(scratch.TopKIDs) < positions*topK || len(scratch.TopKVals) < positions*topK {
		return fmt.Errorf("FP8 CPU experts: invalid top-k scratch positions=%d topK=%d ids=%d vals=%d", positions, topK, len(scratch.TopKIDs), len(scratch.TopKVals))
	}

	for i := range scratch.MoeOut {
		scratch.MoeOut[i] = 0
	}

	// Pre-norm all positions (parallel)
	normedRows := make([]float32, positions*hiddenSize)
	for pos := 0; pos < positions; pos++ {
		resRow := scratch.Residual[pos*hiddenSize : (pos+1)*hiddenSize]
		dst := normedRows[pos*hiddenSize : (pos+1)*hiddenSize]
		copy(dst, resRow)
		if !simd.RMSNormTo(dst, preNorm2, 1e-6) {
			return fmt.Errorf("pre_norm_2 rejected")
		}
	}

	// Collect unique experts and which positions use them
	type posWeight struct {
		pos int
		w   float32
	}
	expertUsers := make(map[int][]posWeight)
	for pos := 0; pos < positions; pos++ {
		for k := 0; k < topK; k++ {
			eid := scratch.TopKIDs[pos*topK+k]
			if eid >= 0 && eid < idx.NumExperts {
				expertUsers[eid] = append(expertUsers[eid], posWeight{pos, scratch.TopKVals[pos*topK+k]})
			}
		}
	}

	layerExperts := idx.entries[op.Layer]
	if len(layerExperts) < idx.NumExperts {
		return fmt.Errorf("FP8 CPU experts: layer %d has %d experts, want %d", op.Layer, len(layerExperts), idx.NumExperts)
	}
	intermediate := 0
	for eid, ep := range layerExperts[:idx.NumExperts] {
		if err := ep.gate.Validate(); err != nil {
			return fmt.Errorf("FP8 CPU experts: layer %d expert %d gate: %w", op.Layer, eid, err)
		}
		if err := ep.up.Validate(); err != nil {
			return fmt.Errorf("FP8 CPU experts: layer %d expert %d up: %w", op.Layer, eid, err)
		}
		if err := ep.down.Validate(); err != nil {
			return fmt.Errorf("FP8 CPU experts: layer %d expert %d down: %w", op.Layer, eid, err)
		}
		if ep.gate.InDim != hiddenSize || ep.up.InDim != hiddenSize || ep.up.OutDim != ep.gate.OutDim || ep.down.InDim != ep.gate.OutDim || ep.down.OutDim != hiddenSize {
			return fmt.Errorf("FP8 CPU experts: layer %d expert %d shape mismatch gate=[%d,%d] up=[%d,%d] down=[%d,%d] hidden=%d", op.Layer, eid, ep.gate.OutDim, ep.gate.InDim, ep.up.OutDim, ep.up.InDim, ep.down.OutDim, ep.down.InDim, hiddenSize)
		}
		if intermediate == 0 {
			intermediate = ep.gate.OutDim
		} else if intermediate != ep.gate.OutDim {
			return fmt.Errorf("FP8 CPU experts: layer %d expert %d intermediate=%d want %d", op.Layer, eid, ep.gate.OutDim, intermediate)
		}
	}

	// Use all available CPU threads
	numWorkers := runtime.NumCPU()
	if numWorkers > len(expertUsers) {
		numWorkers = len(expertUsers)
	}
	// Sort by descending batch size, then greedily assign experts to the
	// currently-lightest worker. This avoids random map-order imbalance when a
	// few experts own many more canvas positions than the rest.
	expertIDs := make([]int, 0, len(expertUsers))
	for eid := range expertUsers {
		expertIDs = append(expertIDs, eid)
	}
	sort.Slice(expertIDs, func(i, j int) bool {
		return len(expertUsers[expertIDs[i]]) > len(expertUsers[expertIDs[j]])
	})
	workerExpertIDs := make([][]int, numWorkers)
	workerLoads := make([]int, numWorkers)
	for _, eid := range expertIDs {
		best := 0
		for w := 1; w < numWorkers; w++ {
			if workerLoads[w] < workerLoads[best] {
				best = w
			}
		}
		workerExpertIDs[best] = append(workerExpertIDs[best], eid)
		workerLoads[best] += len(expertUsers[eid])
	}

	// Each worker accumulates into its own output buffer to avoid contention
	workerOuts := make([][]float32, numWorkers)
	for w := 0; w < numWorkers; w++ {
		workerOuts[w] = make([]float32, len(scratch.MoeOut))
	}

	var wg sync.WaitGroup
	var errOnce sync.Once
	var firstErr error

	for w := 0; w < numWorkers; w++ {
		w := w
		idsForWorker := workerExpertIDs[w]
		if len(idsForWorker) == 0 {
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			out := workerOuts[w]

			// Find max batch size for this worker's experts
			maxBatch := 0
			for _, eid := range idsForWorker {
				if n := len(expertUsers[eid]); n > maxBatch {
					maxBatch = n
				}
			}

			// Pre-allocate worker scratch (reused across experts)
			var ws expertWorkerScratch
			if err := ws.ensure(maxBatch, hiddenSize, intermediate); err != nil {
				errOnce.Do(func() { firstErr = err })
				return
			}

			for _, eid := range idsForWorker {
				ep := layerExperts[eid]
				users := expertUsers[eid]
				nPos := len(users)

				// Gather input rows for this expert
				batchIn := ws.batchIn[:nPos*hiddenSize]
				for i, u := range users {
					copy(batchIn[i*hiddenSize:(i+1)*hiddenSize], normedRows[u.pos*hiddenSize:(u.pos+1)*hiddenSize])
				}

				// gate[nPos, 704] = batchIn[nPos, 2816] × W_gate^T
				batchGate := ws.batchGate[:nPos*intermediate]
				gateInput := batchIn
				if diffusionGemmaFP8DynamicActivationEnabled() {
					gateInput = ws.xq[:nPos*hiddenSize]
					quantizeDynamicTokenBatch(gateInput, batchIn, nPos, hiddenSize)
				}
				if err := ep.gate.BatchGemvToBuf(gateInput, batchGate, nPos, ws.wf32); err != nil {
					errOnce.Do(func() { firstErr = fmt.Errorf("expert %d gate: %w", eid, err) })
					return
				}

				// up[nPos, 704] = batchIn[nPos, 2816] × W_up^T
				batchUp := ws.batchUp[:nPos*intermediate]
				if err := ep.up.BatchGemvToBuf(gateInput, batchUp, nPos, ws.wf32); err != nil {
					errOnce.Do(func() { firstErr = fmt.Errorf("expert %d up: %w", eid, err) })
					return
				}

				// Activation: GELU(gate) * up, per position
				batchAct := ws.batchAct[:nPos*intermediate]
				for i := 0; i < nPos; i++ {
					g := batchGate[i*intermediate : (i+1)*intermediate]
					u := batchUp[i*intermediate : (i+1)*intermediate]
					a := batchAct[i*intermediate : (i+1)*intermediate]
					if !simd.GELUExactMulTo(a, g, u) {
						errOnce.Do(func() { firstErr = fmt.Errorf("expert activation rejected") })
						return
					}
				}

				// down[nPos, 2816] = batchAct[nPos, 704] × W_down^T
				batchDown := ws.batchDown[:nPos*hiddenSize]
				downInput := batchAct
				if diffusionGemmaFP8DynamicActivationEnabled() {
					downInput = ws.xq[:nPos*intermediate]
					quantizeDynamicTokenBatch(downInput, batchAct, nPos, intermediate)
				}
				if err := ep.down.BatchGemvToBuf(downInput, batchDown, nPos, ws.wf32); err != nil {
					errOnce.Do(func() { firstErr = fmt.Errorf("expert %d down: %w", eid, err) })
					return
				}

				// Scatter weighted outputs
				for i, u := range users {
					expertOut := batchDown[i*hiddenSize : (i+1)*hiddenSize]
					dst := out[u.pos*hiddenSize : (u.pos+1)*hiddenSize]
					simd.Saxpy(u.w, expertOut, dst)
				}
			}
		}()
	}
	wg.Wait()
	if firstErr != nil {
		return firstErr
	}

	// Merge worker outputs using SIMD saxpy (y += 1.0 * x)
	for w := 0; w < numWorkers; w++ {
		simd.Saxpy(1.0, workerOuts[w], scratch.MoeOut)
	}

	postNorm2, err := loadFloatVector(weights, lb.PostFFNLayerNorm2)
	if err != nil {
		return err
	}
	for off := 0; off < len(scratch.MoeOut); off += hiddenSize {
		if !simd.RMSNormTo(scratch.MoeOut[off:off+hiddenSize], postNorm2, 1e-6) {
			return fmt.Errorf("post_norm_2 rejected")
		}
	}
	return nil
}
