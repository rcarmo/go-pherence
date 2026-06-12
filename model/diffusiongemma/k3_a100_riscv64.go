//go:build riscv64

package diffusiongemma

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/rcarmo/go-pherence/backends/spacemit/aicpu/aipool"
	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"github.com/rcarmo/go-pherence/half"
)

func k3A100Q8Enabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_Q8")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func k3Threads() int {
	if s := strings.TrimSpace(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_THREADS")); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			return n
		}
	}
	return 8
}

func k3A100Workers() int {
	if s := strings.TrimSpace(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_WORKERS")); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			if n > 8 {
				n = 8
			}
			return n
		}
	}
	return 6
}

var (
	k3A100PoolOnce sync.Once
	k3A100Pool     *aipool.AIWorkerPool
	k3Q80Cache     sync.Map // k3Q80CacheKey -> ime2.Q80x32
)

type k3Q80CacheKey struct {
	weights *TextWeights
	name    string
}

func k3EvictQ80Tensor(weights *TextWeights, name string) bool {
	if weights == nil || name == "" {
		return false
	}
	key := k3Q80CacheKey{weights: weights, name: name}
	_, ok := k3Q80Cache.Load(key)
	if ok {
		k3Q80Cache.Delete(key)
	}
	return ok
}

func k3EvictQ80Layer(weights *TextWeights, layer int) int {
	if weights == nil || layer < 0 {
		return 0
	}
	prefix := fmt.Sprintf("model.decoder.layers.%d.", layer)
	evicted := 0
	k3Q80Cache.Range(func(key, _ any) bool {
		k, ok := key.(k3Q80CacheKey)
		if ok && k.weights == weights && strings.HasPrefix(k.name, prefix) {
			k3Q80Cache.Delete(key)
			evicted++
		}
		return true
	})
	return evicted
}

func k3ClearQ80CacheForWeights(weights *TextWeights) {
	if weights == nil {
		return
	}
	k3Q80Cache.Range(func(key, _ any) bool {
		if k, ok := key.(k3Q80CacheKey); ok && k.weights == weights {
			k3Q80Cache.Delete(key)
		}
		return true
	})
}

func k3A100WorkerPool() *aipool.AIWorkerPool {
	k3A100PoolOnce.Do(func() {
		n := k3A100Workers()
		needP := k3Threads() + n
		if needP > 16 {
			needP = 16
		}
		if runtime.GOMAXPROCS(0) < needP {
			runtime.GOMAXPROCS(needP)
		}
		k3A100Pool = aipool.NewAIWorkerPool(n)
	})
	return k3A100Pool
}

func k3GemmRowsQ80(out, x []float32, batch int, weights *TextWeights, binding *TensorBinding) (bool, error) {
	if !k3Enabled() || !k3A100Q8Enabled() || batch <= 0 || binding == nil {
		return false, nil
	}
	wq, ok, err := k3Q80ForBinding(weights, binding)
	if err != nil || !ok {
		return ok, err
	}
	if len(x) < batch*wq.K || len(out) < batch*wq.M {
		return true, fmt.Errorf("DiffusionGemma K3 A100 GEMM %s invalid buffers x=%d/%d out=%d/%d", binding.Name, len(x), batch*wq.K, len(out), batch*wq.M)
	}
	if aipool.GemmQ80x32AIPooledX100Pack(x[:batch*wq.K], batch, wq.K, wq, out[:batch*wq.M], k3A100WorkerPool()) {
		return true, nil
	}
	return false, nil
}

func k3Gemm2RowsQ80(outA, outB, x []float32, batch int, weights *TextWeights, bindingA, bindingB *TensorBinding) (bool, error) {
	return k3GemmManyRowsQ80([][]float32{outA, outB}, x, batch, weights, []*TensorBinding{bindingA, bindingB})
}

func k3GemmManyRowsQ80(outs [][]float32, x []float32, batch int, weights *TextWeights, bindings []*TensorBinding) (bool, error) {
	if !k3Enabled() || !k3A100Q8Enabled() || batch <= 0 || len(outs) == 0 || len(outs) != len(bindings) {
		return false, nil
	}
	packed := make([]ime2.Q80x32, len(bindings))
	k := 0
	for i, binding := range bindings {
		if binding == nil {
			return false, nil
		}
		wq, ok, err := k3Q80ForBinding(weights, binding)
		if err != nil || !ok {
			return ok, err
		}
		if i == 0 {
			k = wq.K
		} else if wq.K != k {
			return false, nil
		}
		if len(outs[i]) < batch*wq.M {
			return true, fmt.Errorf("DiffusionGemma K3 A100 multi GEMM %s invalid out=%d/%d", binding.Name, len(outs[i]), batch*wq.M)
		}
		packed[i] = wq
	}
	if len(x) < batch*k {
		return true, fmt.Errorf("DiffusionGemma K3 A100 multi GEMM invalid x=%d/%d", len(x), batch*k)
	}
	if aipool.GemmManyQ80x32AIPooledX100PackSameInput(x[:batch*k], batch, k, packed, outs, k3A100WorkerPool()) {
		return true, nil
	}
	return false, nil
}

func k3Q80ForBinding(weights *TextWeights, binding *TensorBinding) (ime2.Q80x32, bool, error) {
	if weights == nil || binding == nil || len(binding.Shape) != 2 {
		return ime2.Q80x32{}, false, nil
	}
	rows, cols := binding.Shape[0], binding.Shape[1]
	if rows <= 0 || cols <= 0 || rows%32 != 0 || cols%32 != 0 {
		return ime2.Q80x32{M: rows, K: cols}, false, nil
	}
	key := k3Q80CacheKey{weights: weights, name: binding.Name}
	if v, ok := k3Q80Cache.Load(key); ok {
		return v.(ime2.Q80x32), true, nil
	}
	raw, dtype, shape, err := weights.RawTensor(binding.Name)
	if err != nil {
		return ime2.Q80x32{}, true, err
	}
	if len(shape) != 2 || shape[0] != rows || shape[1] != cols {
		return ime2.Q80x32{}, true, fmt.Errorf("DiffusionGemma K3 Q80 tensor %s shape %v want [%d %d]", binding.Name, shape, rows, cols)
	}
	var q ime2.Q80x32
	switch dtype {
	case "F8_E4M3", "F8_E4M3FN":
		scales, err := loadK3WeightScales(weights, binding.Name, rows)
		if err != nil {
			return ime2.Q80x32{}, true, err
		}
		q = packK3FP8ToQ80x32RowScale(raw, scales, rows, cols)
	default:
		t, err := weights.CachedFloatTensor(binding.Name)
		if err != nil {
			return ime2.Q80x32{}, true, err
		}
		q = ime2.PackF32ToQ80x32RowScale(rows, cols, t.Data)
	}
	if !q.Valid {
		return q, false, nil
	}
	k3Q80Cache.Store(key, q)
	return q, true, nil
}

func loadK3WeightScales(weights *TextWeights, weightName string, rows int) ([]float32, error) {
	scaleName := weightName + "_scale"
	if strings.HasSuffix(weightName, ".weight") {
		scaleName = strings.TrimSuffix(weightName, ".weight") + ".weight_scale"
	}
	raw, dtype, shape, err := weights.RawTensor(scaleName)
	if err != nil {
		return nil, fmt.Errorf("DiffusionGemma K3 Q80 scale %s: %w", scaleName, err)
	}
	n := 1
	for _, d := range shape {
		if d <= 0 {
			return nil, fmt.Errorf("DiffusionGemma K3 Q80 scale %s invalid shape %v", scaleName, shape)
		}
		n *= d
	}
	if n != 1 && n != rows {
		return nil, fmt.Errorf("DiffusionGemma K3 Q80 scale %s shape %v gives %d values, want 1 or %d", scaleName, shape, n, rows)
	}
	scales := make([]float32, n)
	if err := decodeFloatRowTo(scales, raw, dtype); err != nil {
		return nil, err
	}
	return scales, nil
}

func packK3FP8ToQ80x32RowScale(raw []byte, scales []float32, rows, cols int) ime2.Q80x32 {
	if rows%32 != 0 || cols%32 != 0 || len(raw) < rows*cols || len(scales) == 0 {
		return ime2.Q80x32{M: rows, K: cols}
	}
	groups, subs := rows/32, cols/32
	out := make([]byte, groups*subs*ime2.K3I8I8BTileBytes)
	nw := k3Threads()
	if nw > groups {
		nw = groups
	}
	if nw < 1 {
		nw = 1
	}
	var wg sync.WaitGroup
	wg.Add(nw)
	for wid := 0; wid < nw; wid++ {
		g0 := wid * groups / nw
		g1 := (wid + 1) * groups / nw
		go func(g0, g1 int) {
			defer wg.Done()
			rowBuf := make([]float32, cols)
			for g := g0; g < g1; g++ {
				for rr := 0; rr < 32; rr++ {
					row := g*32 + rr
					scale := scales[0]
					if len(scales) != 1 {
						scale = scales[row]
					}
					base := row * cols
					maxAbs := float32(0)
					for k := 0; k < cols; k++ {
						v := fp8DecodeE4M3(raw[base+k]) * scale
						rowBuf[k] = v
						av := float32(math.Abs(float64(v)))
						if av > maxAbs {
							maxAbs = av
						}
					}
					d := float32(0)
					if maxAbs != 0 {
						d = maxAbs / 127.0
					}
					inv := float32(0)
					if d != 0 {
						inv = 1 / d
					}
					for sb := 0; sb < subs; sb++ {
						block := out[(g*subs+sb)*ime2.K3I8I8BTileBytes:]
						binary.LittleEndian.PutUint16(block[rr*2:], half.F32ToF16(d))
						qs := block[64 : 64+1024]
						for k := 0; k < 32; k++ {
							q := int(math.Round(float64(rowBuf[sb*32+k] * inv)))
							if q > 127 {
								q = 127
							}
							if q < -128 {
								q = -128
							}
							qs[rr*32+k] = byte(int8(q))
						}
					}
				}
			}
		}(g0, g1)
	}
	wg.Wait()
	return ime2.Q80x32{M: rows, K: cols, BData: out, Valid: true}
}

func k3Q80ForTensorName(weights *TextWeights, name string) (ime2.Q80x32, bool, error) {
	if weights == nil || name == "" {
		return ime2.Q80x32{}, false, nil
	}
	key := k3Q80CacheKey{weights: weights, name: name}
	if v, ok := k3Q80Cache.Load(key); ok {
		return v.(ime2.Q80x32), true, nil
	}
	raw, dtype, shape, err := weights.RawTensor(name)
	if err != nil {
		return ime2.Q80x32{}, true, err
	}
	if len(shape) != 2 || shape[0] <= 0 || shape[1] <= 0 || shape[0]%32 != 0 || shape[1]%32 != 0 {
		return ime2.Q80x32{}, false, nil
	}
	rows, cols := shape[0], shape[1]
	var q ime2.Q80x32
	switch dtype {
	case "F8_E4M3", "F8_E4M3FN":
		scales, err := loadK3WeightScales(weights, name, rows)
		if err != nil {
			return ime2.Q80x32{}, true, err
		}
		q = packK3FP8ToQ80x32RowScale(raw, scales, rows, cols)
	default:
		t, err := weights.CachedFloatTensor(name)
		if err != nil {
			return ime2.Q80x32{}, true, err
		}
		q = ime2.PackF32ToQ80x32RowScale(rows, cols, t.Data)
	}
	if !q.Valid {
		return q, false, nil
	}
	k3Q80Cache.Store(key, q)
	return q, true, nil
}

func k3GemmRowsQ80Name(out, x []float32, batch int, weights *TextWeights, name string) (bool, error) {
	if !k3Enabled() || !k3A100Q8Enabled() || batch <= 0 || name == "" {
		return false, nil
	}
	wq, ok, err := k3Q80ForTensorName(weights, name)
	if err != nil || !ok {
		return ok, err
	}
	if len(x) < batch*wq.K || len(out) < batch*wq.M {
		return true, fmt.Errorf("DiffusionGemma K3 A100 GEMM %s invalid buffers x=%d/%d out=%d/%d", name, len(x), batch*wq.K, len(out), batch*wq.M)
	}
	if aipool.GemmQ80x32AIPooledX100Pack(x[:batch*wq.K], batch, wq.K, wq, out[:batch*wq.M], k3A100WorkerPool()) {
		return true, nil
	}
	return false, nil
}

func k3Gemm2RowsQ80Names(outA, outB, x []float32, batch int, weights *TextWeights, nameA, nameB string) (bool, error) {
	if !k3Enabled() || !k3A100Q8Enabled() || batch <= 0 || nameA == "" || nameB == "" {
		return false, nil
	}
	wA, okA, err := k3Q80ForTensorName(weights, nameA)
	if err != nil || !okA {
		return okA, err
	}
	wB, okB, err := k3Q80ForTensorName(weights, nameB)
	if err != nil || !okB {
		return okB, err
	}
	if wA.K != wB.K || len(x) < batch*wA.K || len(outA) < batch*wA.M || len(outB) < batch*wB.M {
		return true, fmt.Errorf("DiffusionGemma K3 A100 dual GEMM invalid names=%s,%s", nameA, nameB)
	}
	if aipool.GemmManyQ80x32AIPooledX100PackSameInput(x[:batch*wA.K], batch, wA.K, []ime2.Q80x32{wA, wB}, [][]float32{outA[:batch*wA.M], outB[:batch*wB.M]}, k3A100WorkerPool()) {
		return true, nil
	}
	return false, nil
}

// PreloadLayerQ80 pre-packs K3 A100 row-scale Q80x32 weights for one layer.
// Dense projections are always prepacked; per-expert tensors are only included
// when includeExperts is true because they dominate memory.
func (w *TextWeights) PreloadLayerQ80(layer int, includeExperts bool) (int, error) {
	if w == nil {
		return 0, fmt.Errorf("nil DiffusionGemma text weights")
	}
	fp := w.ForwardPlan()
	if layer < 0 || layer >= len(fp.Layers) {
		return 0, fmt.Errorf("DiffusionGemma layer %d outside [0,%d)", layer, len(fp.Layers))
	}
	lb := fp.Layers[layer]
	count := 0
	for _, b := range []*TensorBinding{lb.QProj, lb.KProj, lb.VProj, lb.OProj, lb.MLPGateProj, lb.MLPUpProj, lb.MLPDownProj, lb.RouterProj} {
		if b == nil {
			continue
		}
		if _, ok, err := k3Q80ForBinding(w, b); err != nil {
			return count, err
		} else if ok {
			count++
		}
	}
	if includeExperts {
		hiddenSize := 0
		if lb.PreFFNLayerNorm2 != nil && len(lb.PreFFNLayerNorm2.Shape) == 1 {
			hiddenSize = lb.PreFFNLayerNorm2.Shape[0]
		} else if fp.Globals.EmbedTokens != nil && len(fp.Globals.EmbedTokens.Shape) == 2 {
			hiddenSize = fp.Globals.EmbedTokens.Shape[1]
		}
		if hiddenSize <= 0 {
			return count, fmt.Errorf("DiffusionGemma layer %d cannot infer hidden size for expert Q80 prewarm", layer)
		}
		layout, err := expertLayoutForLayer(w, lb, hiddenSize)
		if err != nil {
			return count, err
		}
		if layout.fused {
			for _, b := range []*TensorBinding{lb.ExpertsGateUpProj, lb.ExpertsDownProj} {
				if b == nil {
					continue
				}
				if _, ok, err := k3Q80ForBinding(w, b); err != nil {
					return count, err
				} else if ok {
					count++
				}
			}
		} else {
			names := make([]string, 0, layout.nExperts*3)
			for expertID := 0; expertID < layout.nExperts; expertID++ {
				for _, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
					names = append(names, perExpertTensorName(layout.layerPrefix, expertID, proj))
				}
			}
			if ok, err := k3PrepackExpertQ80Names(w, names); err != nil {
				return count, err
			} else if ok {
				count += len(names)
			}
		}
	}
	return count, nil
}

func (w *TextWeights) PreloadLayerRangeQ80(start, count int, includeExperts bool) (int, error) {
	if count < 0 {
		return 0, fmt.Errorf("DiffusionGemma negative Q80 preload count %d", count)
	}
	if w != nil && start == 0 && count > w.q80ResidentLayerPrefix {
		w.q80ResidentLayerPrefix = count
	}
	total := 0
	for i := 0; i < count; i++ {
		n, err := w.PreloadLayerQ80(start+i, includeExperts)
		if err != nil {
			return total, err
		}
		total += n
	}
	return total, nil
}
