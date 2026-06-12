//go:build riscv64

package diffusiongemma

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

type k3ExpertAssignment struct {
	pos    int
	weight float32
}

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
	workers := k3ExpertPrepackWorkers(len(names))
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
	for _, name := range names {
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
	assignments := map[int][]k3ExpertAssignment{}
	for pos := 0; pos < positions; pos++ {
		for k := 0; k < topK; k++ {
			expertID := topIDs[pos*topK+k]
			if expertID < 0 || expertID >= layout.nExperts {
				continue
			}
			assignments[expertID] = append(assignments[expertID], k3ExpertAssignment{pos: pos, weight: topVals[pos*topK+k]})
		}
	}
	assignmentCount := 0
	for _, rows := range assignments {
		assignmentCount += len(rows)
	}
	prepackNames := make([]string, 0, len(assignments)*3)
	for expertID := range assignments {
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
	for expertID, rows := range assignments {
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
