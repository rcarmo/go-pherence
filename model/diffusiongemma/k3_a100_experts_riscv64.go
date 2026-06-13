//go:build riscv64

package diffusiongemma

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func k3ExpertPrepackWorkers(tasks int) int {
	if tasks <= 1 {
		return tasks
	}
	workers := k3Threads()
	if s := strings.TrimSpace(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_EXPERT_PREPACK_WORKERS")); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			workers = n
		}
	}
	if workers > tasks {
		workers = tasks
	}
	return workers
}

func k3PrepackExpertQ80Names(weights *TextWeights, names []string) (bool, error) {
	if len(names) == 0 {
		return true, nil
	}
	pending := make([]string, 0, len(names))
	seen := make(map[string]struct{}, len(names))
	for _, name := range names {
		if name == "" {
			continue
		}
		if _, ok := seen[name]; ok {
			continue
		}
		seen[name] = struct{}{}
		if k3Q80CachedForTensorName(weights, name) {
			continue
		}
		pending = append(pending, name)
	}
	if len(pending) == 0 {
		return true, nil
	}
	workers := k3ExpertPrepackWorkers(len(pending))
	jobs := make(chan string)
	var wg sync.WaitGroup
	var mu sync.Mutex
	okAll := true
	var firstErr error
	wg.Add(workers)
	for wid := 0; wid < workers; wid++ {
		go func() {
			defer wg.Done()
			for name := range jobs {
				_, ok, err := k3Q80ForTensorName(weights, name)
				if err != nil || !ok {
					mu.Lock()
					okAll = okAll && ok
					if firstErr == nil {
						firstErr = err
					}
					mu.Unlock()
				}
			}
		}()
	}
	for _, name := range pending {
		jobs <- name
	}
	close(jobs)
	wg.Wait()
	return okAll, firstErr
}

func k3RunPerExpertA100(weights *TextWeights, lb TextLayerBindings, layout expertWeightLayout, scratch ForwardScratch, preNorm2 []float32, hiddenSize, positions, topK int) (bool, error) {
	return k3RunPerExpertRowsA100(weights, layout, scratch.Residual, scratch.TopKIDs, scratch.TopKVals, scratch.MoeOut, preNorm2, hiddenSize, positions, topK)
}

func k3RunPerExpertRowsA100(weights *TextWeights, layout expertWeightLayout, residual []float32, topIDs []int, topVals, moeOut []float32, preNorm2 []float32, hiddenSize, positions, topK int) (bool, error) {
	if !k3Enabled() || !k3A100Q8Enabled() || layout.fused || layout.layerPrefix == "" || positions <= 0 || topK <= 0 {
		return false, nil
	}
	if len(residual) < positions*hiddenSize || len(topIDs) < positions*topK || len(topVals) < positions*topK || len(moeOut) < positions*hiddenSize {
		return true, fmt.Errorf("DiffusionGemma K3 A100 expert invalid buffers")
	}
	normed := make([]float32, positions*hiddenSize)
	for pos := 0; pos < positions; pos++ {
		row := normed[pos*hiddenSize : (pos+1)*hiddenSize]
		copy(row, residual[pos*hiddenSize:(pos+1)*hiddenSize])
		if !simd.RMSNormTo(row, preNorm2, 1e-6) {
			return true, fmt.Errorf("DiffusionGemma K3 A100 expert pre_norm_2 rejected")
		}
	}
	assignments := map[int][]expertAssignment{}
	for pos := 0; pos < positions; pos++ {
		for k := 0; k < topK; k++ {
			expertID := topIDs[pos*topK+k]
			if expertID < 0 || expertID >= layout.nExperts {
				continue
			}
			assignments[expertID] = append(assignments[expertID], expertAssignment{pos: pos, weight: topVals[pos*topK+k]})
		}
	}
	expertIDs := make([]int, 0, len(assignments))
	assignmentCount := 0
	for expertID, rows := range assignments {
		expertIDs = append(expertIDs, expertID)
		assignmentCount += len(rows)
	}
	sort.Ints(expertIDs)
	prepackNames := make([]string, 0, len(assignments)*3)
	for _, expertID := range expertIDs {
		for _, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
			prepackNames = append(prepackNames, perExpertTensorName(layout.layerPrefix, expertID, proj))
		}
	}
	prepackStart := time.Now()
	if ok, err := k3PrepackExpertQ80Names(weights, prepackNames); err != nil || !ok {
		return ok, err
	}
	prepackElapsed := time.Since(prepackStart)
	ensure := func(buf []float32, n int) []float32 {
		if cap(buf) < n {
			return make([]float32, n)
		}
		return buf[:n]
	}
	var xBuf, gateBuf, upBuf, actBuf, downBuf []float32
	var gateUpElapsed, actElapsed, downElapsed, accumElapsed time.Duration
	for _, expertID := range expertIDs {
		rows := assignments[expertID]
		batch := len(rows)
		if batch == 0 {
			continue
		}
		x := ensure(xBuf, batch*hiddenSize)
		xBuf = x
		for i, a := range rows {
			copy(x[i*hiddenSize:(i+1)*hiddenSize], normed[a.pos*hiddenSize:(a.pos+1)*hiddenSize])
		}
		gate := ensure(gateBuf, batch*layout.intermediate)
		gateBuf = gate
		up := ensure(upBuf, batch*layout.intermediate)
		upBuf = up
		gateName := perExpertTensorName(layout.layerPrefix, expertID, "gate_proj")
		upName := perExpertTensorName(layout.layerPrefix, expertID, "up_proj")
		phaseStart := time.Now()
		done, err := k3Gemm2RowsQ80Names(gate, up, x, batch, weights, gateName, upName)
		gateUpElapsed += time.Since(phaseStart)
		if err != nil || !done {
			return done, err
		}
		act := ensure(actBuf, batch*layout.intermediate)
		actBuf = act
		phaseStart = time.Now()
		for i := 0; i < batch; i++ {
			if !simd.GELUTanhMulTo(act[i*layout.intermediate:(i+1)*layout.intermediate], gate[i*layout.intermediate:(i+1)*layout.intermediate], up[i*layout.intermediate:(i+1)*layout.intermediate]) {
				return true, fmt.Errorf("DiffusionGemma K3 A100 expert activation rejected")
			}
		}
		actElapsed += time.Since(phaseStart)
		down := ensure(downBuf, batch*hiddenSize)
		downBuf = down
		downName := perExpertTensorName(layout.layerPrefix, expertID, "down_proj")
		phaseStart = time.Now()
		done, err = k3GemmRowsQ80Name(down, act, batch, weights, downName)
		downElapsed += time.Since(phaseStart)
		if err != nil || !done {
			return done, err
		}
		phaseStart = time.Now()
		for i, a := range rows {
			dst := moeOut[a.pos*hiddenSize : (a.pos+1)*hiddenSize]
			src := down[i*hiddenSize : (i+1)*hiddenSize]
			k3SaxpyV(a.weight, src, dst)
		}
		accumElapsed += time.Since(phaseStart)
	}
	if diffusionGemmaTimingEnabled() {
		fmt.Fprintf(os.Stderr, "timing diffusiongemma experts unique=%d assignments=%d prepack=%s gate_up=%s act=%s down=%s accum=%s\n", len(assignments), assignmentCount, prepackElapsed.Round(time.Millisecond), gateUpElapsed.Round(time.Millisecond), actElapsed.Round(time.Millisecond), downElapsed.Round(time.Millisecond), accumElapsed.Round(time.Millisecond))
	}
	return true, nil
}

type k3SelectedExpertPrefetch struct {
	layer          int
	includeExperts bool
	ch             chan q80PrefetchResult
}

func k3StartSelectedExpertQ80Prefetch(weights *TextWeights, lb TextLayerBindings, scratch ForwardScratch) *k3SelectedExpertPrefetch {
	if weights == nil || !k3Enabled() || !k3A100Q8Enabled() || !k3Q80SelectedPrefetchEnabled() || !lb.HasPerExpertWeights {
		return nil
	}
	hiddenSize := 0
	if lb.PreFFNLayerNorm2 != nil && len(lb.PreFFNLayerNorm2.Shape) == 1 {
		hiddenSize = lb.PreFFNLayerNorm2.Shape[0]
	}
	if hiddenSize <= 0 || len(scratch.Residual)%hiddenSize != 0 {
		return nil
	}
	positions := len(scratch.Residual) / hiddenSize
	topK := scratch.TopKExperts
	if positions <= 0 || topK <= 0 || len(scratch.TopKIDs) < positions*topK || len(scratch.TopKVals) < positions*topK {
		return nil
	}
	layout, err := expertLayoutForLayer(weights, lb, hiddenSize)
	if err != nil || layout.fused || layout.layerPrefix == "" {
		return nil
	}
	seen := map[int]bool{}
	expertIDs := make([]int, 0, positions*topK)
	for pos := 0; pos < positions; pos++ {
		for k := 0; k < topK; k++ {
			expertID := scratch.TopKIDs[pos*topK+k]
			if expertID < 0 || expertID >= layout.nExperts || seen[expertID] {
				continue
			}
			seen[expertID] = true
			expertIDs = append(expertIDs, expertID)
		}
	}
	sort.Ints(expertIDs)
	names := make([]string, 0, len(expertIDs)*3)
	for _, expertID := range expertIDs {
		for _, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
			names = append(names, perExpertTensorName(layout.layerPrefix, expertID, proj))
		}
	}
	if len(names) == 0 {
		return nil
	}
	p := &k3SelectedExpertPrefetch{layer: lb.Layer, includeExperts: true, ch: make(chan q80PrefetchResult, 1)}
	go func() {
		started := time.Now()
		ok, err := k3PrepackExpertQ80Names(weights, names)
		count := 0
		if ok {
			count = len(names)
		}
		p.ch <- q80PrefetchResult{count: count, err: err, elapsed: time.Since(started)}
	}()
	return p
}

func (p *k3SelectedExpertPrefetch) Wait(weights *TextWeights, progress bool) error {
	if p == nil || p.ch == nil {
		return nil
	}
	res := <-p.ch
	if progress {
		fmt.Fprintf(os.Stderr, "timing diffusiongemma selected_expert_prefetch layer=%d tensors=%d elapsed=%s q80_entries=%d q80_bytes=%d\n", p.layer, res.count, res.elapsed.Round(time.Millisecond), weights.Q80CacheEntries(), weights.Q80CacheBytes())
	}
	return res.err
}
