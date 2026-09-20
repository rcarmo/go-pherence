package needle

import (
	"fmt"
	"math"
)

// Needle2 has one softmax over token/layer cells, no per-layer RMS or second
// query pool. Keep its parameter names and fixed probe counts distinct from v3.
func (m *Model) headGeometryV2(kind HeadKind) (k, q, out int, err error) {
	prefix := string(kind) + "_head/"
	switch kind {
	case Contrastive:
		k, out = 4, m.config.ContrastiveDim
		if out == 0 {
			out = 128
		}
	case Confidence:
		k, out = 8, 1
	default:
		return 0, 0, 0, fmt.Errorf("needle: Needle2 supports contrastive and confidence heads only")
	}
	if out < 1 || out > 4096 {
		return 0, 0, 0, fmt.Errorf("needle: invalid Needle2 head output size")
	}
	shapes := map[string][]int{"probes": {k, m.config.DModel}, "proj/kernel": {k * m.config.DModel, out}}
	if kind == Confidence {
		shapes["proj/bias"] = []int{1}
	} else if !m.deployed {
		// Deployment format omits temperature; inference does not consume it.
		shapes["log_temp"] = []int{}
	}
	for name, shape := range shapes {
		p, ok := m.tensors[prefix+name]
		if !ok || len(p.Shape) != len(shape) {
			return 0, 0, 0, fmt.Errorf("needle: missing or invalid %s%s", prefix, name)
		}
		for i, d := range shape {
			if p.Shape[i] != d {
				return 0, 0, 0, fmt.Errorf("needle: invalid %s%s shape", prefix, name)
			}
		}
	}
	return k, 0, out, nil
}

func (e *execution) probeHeadV2(kind HeadKind) *value {
	t := e.t
	d := e.m.config.DModel
	layers := len(e.cells)
	tokens := len(e.keep)
	// concat is token-major: [token,layer,width], matching upstream reshape.
	joined := t.concat(e.cells)
	cells := t.slice(joined, 0, tokens*layers, d)
	prefix := string(kind) + "_head/"
	scores := t.scale(t.mm(e.param(prefix+"probes", -1), cells, true), float32(1/math.Sqrt(float64(d))))
	masked := t.alloc(scores.r, scores.c)
	copy(masked.x, scores.x)
	for row := 0; row < scores.r; row++ {
		for token, keep := range e.keep {
			if !keep {
				for layer := 0; layer < layers; layer++ {
					masked.x[row*scores.c+token*layers+layer] = float32(math.Inf(-1))
				}
			}
		}
	}
	if t.train {
		t.record(func() {
			for row := 0; row < scores.r; row++ {
				for token, keep := range e.keep {
					if keep {
						for layer := 0; layer < layers; layer++ {
							i := row*scores.c + token*layers + layer
							scores.g[i] += masked.g[i]
						}
					}
				}
			}
		})
	}
	pooled := t.mm(t.softmax(masked, false, 0), cells, false)
	flat := t.slice(pooled, 0, 1, pooled.r*d)
	result := t.mm(flat, e.param(prefix+"proj/kernel", -1), false)
	if kind == Contrastive {
		return t.unitHead(result)
	}
	return t.add(result, e.param(prefix+"proj/bias", -1))
}
