//go:build riscv64

package diffusiongemma

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sync"
	"time"

	"github.com/rcarmo/go-pherence/backends/spacemit/aicpu/aipool"
	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"github.com/rcarmo/go-pherence/half"
)

func k3SelfConditioningSoftEmbeddingQ80(out []float32, logits [][]float32, weights *TextWeights, binding *TensorBinding, positions, vocab, hiddenSize int, tempInv float32) (bool, error) {
	if !k3Enabled() || !k3A100Q8Enabled() || weights == nil || binding == nil || positions <= 0 || vocab <= 0 || hiddenSize <= 0 {
		return false, nil
	}
	if len(out) < positions*hiddenSize || len(logits) < positions {
		return true, fmt.Errorf("DiffusionGemma K3 SC invalid buffers out=%d logits=%d positions=%d hidden=%d", len(out), len(logits), positions, hiddenSize)
	}
	if vocab%32 != 0 || hiddenSize%32 != 0 {
		return false, nil
	}
	wq, ok, err := k3Q80TransposedForBinding(weights, binding)
	if err != nil || !ok {
		return ok, err
	}
	if wq.M != hiddenSize || wq.K != vocab {
		return true, fmt.Errorf("DiffusionGemma K3 SC transposed shape M=%d K=%d want M=%d K=%d", wq.M, wq.K, hiddenSize, vocab)
	}
	probs := make([]float32, positions*vocab)
	probStart := time.Now()
	if err := buildSelfConditioningProbRows(probs, logits, positions, vocab, tempInv); err != nil {
		return true, err
	}
	probElapsed := time.Since(probStart)
	gemmStart := time.Now()
	if aipool.GemmQ80x32AIPooledX100Pack(probs, positions, vocab, wq, out[:positions*hiddenSize], k3A100WorkerPool()) {
		if diffusionGemmaTimingEnabled() {
			fmt.Fprintf(os.Stderr, "timing diffusiongemma sc_q80 positions=%d vocab=%d hidden=%d prob=%s gemm=%s\n", positions, vocab, hiddenSize, probElapsed.Round(time.Millisecond), time.Since(gemmStart).Round(time.Millisecond))
		}
		return true, nil
	}
	return false, nil
}

func buildSelfConditioningProbRows(probs []float32, logits [][]float32, positions, vocab int, tempInv float32) error {
	if len(probs) < positions*vocab || len(logits) < positions {
		return fmt.Errorf("DiffusionGemma SC probs shape mismatch")
	}
	if tempInv == 0 {
		tempInv = 1
	}
	for pos := 0; pos < positions; pos++ {
		row := logits[pos]
		if len(row) < vocab {
			return fmt.Errorf("DiffusionGemma SC logits row=%d len=%d want %d", pos, len(row), vocab)
		}
		maxLogit := float32(math.Inf(-1))
		for vocabID := 0; vocabID < vocab; vocabID++ {
			v := row[vocabID]
			if math.IsInf(float64(v), -1) || math.IsNaN(float64(v)) {
				continue
			}
			v *= tempInv
			if v > maxLogit {
				maxLogit = v
			}
		}
		out := probs[pos*vocab : (pos+1)*vocab]
		for i := range out {
			out[i] = 0
		}
		if math.IsInf(float64(maxLogit), -1) {
			continue
		}
		var sum float64
		for vocabID := 0; vocabID < vocab; vocabID++ {
			v := row[vocabID]
			if math.IsInf(float64(v), -1) || math.IsNaN(float64(v)) {
				continue
			}
			sum += math.Exp(float64(v*tempInv - maxLogit))
		}
		if sum <= 0 || math.IsNaN(sum) {
			continue
		}
		inv := 1.0 / sum
		for vocabID := 0; vocabID < vocab; vocabID++ {
			v := row[vocabID]
			if math.IsInf(float64(v), -1) || math.IsNaN(float64(v)) {
				continue
			}
			out[vocabID] = float32(math.Exp(float64(v*tempInv-maxLogit)) * inv)
		}
	}
	return nil
}

func k3PreloadQ80TransposedBinding(weights *TextWeights, binding *TensorBinding) (bool, error) {
	_, ok, err := k3Q80TransposedForBinding(weights, binding)
	return ok, err
}

func k3Q80TransposedForBinding(weights *TextWeights, binding *TensorBinding) (ime2.Q80x32, bool, error) {
	if weights == nil || binding == nil || len(binding.Shape) != 2 {
		return ime2.Q80x32{}, false, nil
	}
	rows, cols := binding.Shape[0], binding.Shape[1]
	if rows <= 0 || cols <= 0 || rows%32 != 0 || cols%32 != 0 {
		return ime2.Q80x32{M: cols, K: rows}, false, nil
	}
	key := k3Q80CacheKey{weights: weights, name: binding.Name + "#transpose"}
	if v, ok := k3Q80Cache.Load(key); ok {
		return v.(ime2.Q80x32), true, nil
	}
	raw, dtype, shape, err := weights.RawTensor(binding.Name)
	if err != nil {
		return ime2.Q80x32{}, true, err
	}
	if len(shape) != 2 || shape[0] != rows || shape[1] != cols {
		return ime2.Q80x32{}, true, fmt.Errorf("DiffusionGemma K3 Q80 transpose tensor %s shape %v want [%d %d]", binding.Name, shape, rows, cols)
	}
	var scales []float32
	if dtype == "F8_E4M3" || dtype == "F8_E4M3FN" {
		scales, err = loadK3WeightScales(weights, binding.Name, rows)
		if err != nil {
			return ime2.Q80x32{}, true, err
		}
	}
	q := packK3RawTransposeToQ80x32RowScale(raw, dtype, scales, rows, cols)
	if !q.Valid {
		return q, false, nil
	}
	k3Q80Cache.Store(key, q)
	return q, true, nil
}

func packK3RawTransposeToQ80x32RowScale(raw []byte, dtype string, scales []float32, rows, cols int) ime2.Q80x32 {
	m, k := cols, rows
	if rows%32 != 0 || cols%32 != 0 {
		return ime2.Q80x32{M: m, K: k}
	}
	elemSize, ok := diffusionGemmaDTypeSize(dtype)
	if !ok || len(raw) < rows*cols*elemSize {
		return ime2.Q80x32{M: m, K: k}
	}
	groups, subs := m/32, k/32
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
			for g := g0; g < g1; g++ {
				for sb := 0; sb < subs; sb++ {
					block := out[(g*subs+sb)*ime2.K3I8I8BTileBytes:]
					scaleBytes := block[:64]
					qs := block[64 : 64+1024]
					for rr := 0; rr < 32; rr++ {
						hidden := g*32 + rr
						var vals [32]float32
						maxAbs := float32(0)
						for kk := 0; kk < 32; kk++ {
							vocabID := sb*32 + kk
							v := rawTensor2DValue(raw, dtype, scales, rows, cols, vocabID, hidden)
							vals[kk] = v
							av := float32(math.Abs(float64(v)))
							if av > maxAbs {
								maxAbs = av
							}
						}
						d := float32(0)
						if maxAbs != 0 {
							d = maxAbs / 127.0
						}
						binary.LittleEndian.PutUint16(scaleBytes[rr*2:], half.F32ToF16(d))
						inv := float32(0)
						if d != 0 {
							inv = 1 / d
						}
						for kk, v := range vals {
							q := int(math.Round(float64(v * inv)))
							if q > 127 {
								q = 127
							}
							if q < -128 {
								q = -128
							}
							qs[rr*32+kk] = byte(int8(q))
						}
					}
				}
			}
		}(g0, g1)
	}
	wg.Wait()
	return ime2.Q80x32{M: m, K: k, BData: out, Valid: true}
}

func rawTensor2DValue(raw []byte, dtype string, scales []float32, rows, cols, row, col int) float32 {
	idx := row*cols + col
	switch dtype {
	case "F32":
		return math.Float32frombits(binary.LittleEndian.Uint32(raw[idx*4:]))
	case "BF16":
		return math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[idx*2:])) << 16)
	case "F16":
		return diffusionGemmaF16ToF32(binary.LittleEndian.Uint16(raw[idx*2:]))
	case "F8_E4M3", "F8_E4M3FN":
		scale := float32(1)
		if len(scales) == 1 {
			scale = scales[0]
		} else if len(scales) == rows {
			scale = scales[row]
		}
		return diffusionGemmaFP8E4M3Table[raw[idx]] * scale
	default:
		return 0
	}
}
