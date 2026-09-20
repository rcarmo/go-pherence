package needle

import (
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

func ladderLayerOrder(numLayers int) ([]int, error) {
	if numLayers < 1 {
		return nil, fmt.Errorf("needle: models require at least one layer")
	}
	if numLayers == 1 {
		return []int{0}, nil
	}
	selected := []int{0, numLayers - 1}
	order := append([]int(nil), selected...)
	for len(order) < numLayers {
		slices.Sort(selected)
		bestGap, bestLeft, bestRight := -1, 0, 0
		for i := 0; i+1 < len(selected); i++ {
			left, right := selected[i], selected[i+1]
			gap := right - left
			if gap <= 1 {
				continue
			}
			if gap > bestGap || (gap == bestGap && left < bestLeft) {
				bestGap, bestLeft, bestRight = gap, left, right
			}
		}
		if bestGap <= 1 {
			return nil, fmt.Errorf("needle: failed to construct ladder order for %d layers", numLayers)
		}
		candidate := (bestLeft + bestRight) / 2
		selected = append(selected, candidate)
		order = append(order, candidate)
	}
	return order, nil
}

func ladderOrderForConfig(c Config) ([]int, error) {
	if len(c.LadderOrder) > 0 {
		if len(c.LadderOrder) != c.Layers {
			return nil, fmt.Errorf("needle: ladder_order %v is not a permutation of %d blocks", c.LadderOrder, c.Layers)
		}
		seen := make([]bool, c.Layers)
		for _, layer := range c.LadderOrder {
			if layer < 0 || layer >= c.Layers || seen[layer] {
				return nil, fmt.Errorf("needle: ladder_order %v is not a permutation of %d blocks", c.LadderOrder, c.Layers)
			}
			seen[layer] = true
		}
		return append([]int(nil), c.LadderOrder...), nil
	}
	return ladderLayerOrder(c.Layers)
}

func ladderLayerIndices(c Config, depth int) ([]int, []int, error) {
	if depth < 2 || depth > c.Layers {
		return nil, nil, fmt.Errorf("needle: ladder depth must be in [2, %d], got %d", c.Layers, depth)
	}
	order, err := ladderOrderForConfig(c)
	if err != nil {
		return nil, nil, err
	}
	selected := append([]int(nil), order[:depth]...)
	slices.Sort(selected)
	return selected, order, nil
}

func cloneCheckpointTensor(t checkpoint.Tensor) checkpoint.Tensor {
	return checkpoint.Tensor{Shape: append([]int(nil), t.Shape...), Data: append([]float32(nil), t.Data...)}
}

func tensorSliceAxis(t checkpoint.Tensor, axis int, indices []int) (checkpoint.Tensor, error) {
	if axis < 0 || axis >= len(t.Shape) {
		return checkpoint.Tensor{}, fmt.Errorf("axis %d out of range for rank %d", axis, len(t.Shape))
	}
	if len(indices) == 0 {
		return checkpoint.Tensor{}, fmt.Errorf("empty selection")
	}
	dim := t.Shape[axis]
	for _, idx := range indices {
		if idx < 0 || idx >= dim {
			return checkpoint.Tensor{}, fmt.Errorf("index %d out of range for axis %d size %d", idx, axis, dim)
		}
	}
	outer, inner := 1, 1
	for _, d := range t.Shape[:axis] {
		outer *= d
	}
	for _, d := range t.Shape[axis+1:] {
		inner *= d
	}
	shape := append([]int(nil), t.Shape...)
	shape[axis] = len(indices)
	data := make([]float32, outer*len(indices)*inner)
	for o := 0; o < outer; o++ {
		for j, idx := range indices {
			src := (o*dim + idx) * inner
			dst := (o*len(indices) + j) * inner
			copy(data[dst:dst+inner], t.Data[src:src+inner])
		}
	}
	return checkpoint.Tensor{Shape: shape, Data: data}, nil
}

func parseEngramTensorName(name string) (site int, rest string, ok bool) {
	if !strings.HasPrefix(name, "engrams_") {
		return 0, "", false
	}
	tail := strings.TrimPrefix(name, "engrams_")
	slash := strings.IndexByte(tail, '/')
	if slash <= 0 {
		return 0, "", false
	}
	site, err := strconv.Atoi(tail[:slash])
	if err != nil || site < 0 {
		return 0, "", false
	}
	return site, tail[slash:], true
}

func (m *Model) slicedRawConfig(depth int, order, selected []int) (json.RawMessage, error) {
	fields := map[string]json.RawMessage{}
	if len(m.rawConfig) > 0 {
		if err := json.Unmarshal(m.rawConfig, &fields); err != nil {
			return nil, fmt.Errorf("needle: decode raw config: %w", err)
		}
	}
	set := func(key string, v any) error {
		b, err := json.Marshal(v)
		if err != nil {
			return err
		}
		fields[key] = b
		return nil
	}
	remap := make(map[int]int, len(selected))
	for i, layer := range selected {
		remap[layer] = i
	}
	ladder := make([]int, 0, depth)
	for _, layer := range order {
		if idx, ok := remap[layer]; ok {
			ladder = append(ladder, idx)
		}
	}
	global := make([]int, 0, len(m.config.GlobalLayers))
	for _, layer := range m.config.GlobalLayers {
		if idx, ok := remap[layer]; ok {
			global = append(global, idx)
		}
	}
	engram := make([]int, 0, len(m.config.EngramLayers))
	for _, layer := range m.config.EngramLayers {
		if idx, ok := remap[layer]; ok {
			engram = append(engram, idx)
		}
	}
	for _, kv := range []struct {
		key string
		val any
	}{
		{"num_layers", depth},
		{"ladder_order", ladder},
		{"global_layers", global},
		{"engram_layers", engram},
		{"ladder_depths", []int{}},
		{"ladder_sample", false},
	} {
		if err := set(kv.key, kv.val); err != nil {
			return nil, err
		}
	}
	out, err := json.Marshal(fields)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// SliceDepth returns a reduced-depth Needle3 rung with nested ladder ordering.
// Depth must be in [2, num_layers]. Depth == num_layers still returns an
// independent copy with remapped ladder metadata.
func (m *Model) SliceDepth(depth int) (*Model, error) {
	if m == nil {
		return nil, fmt.Errorf("needle: nil model")
	}
	if m.config.Generation != 3 {
		return nil, fmt.Errorf("needle: SliceDepth is only supported for Needle3 checkpoints")
	}
	selected, order, err := ladderLayerIndices(m.config, depth)
	if err != nil {
		return nil, err
	}
	if depth < m.config.Layers {
		for name := range m.tensors {
			if strings.HasPrefix(name, "ab_scales/") {
				return nil, fmt.Errorf("needle: SliceDepth does not support AB scales on reduced-depth checkpoints")
			}
		}
	}
	rows := make([]int, 0, depth+1)
	rows = append(rows, 0)
	for _, layer := range selected {
		rows = append(rows, layer+1)
	}
	selectedSet := make(map[int]bool, len(selected))
	for _, layer := range selected {
		selectedSet[layer] = true
	}
	raw, err := m.slicedRawConfig(depth, order, selected)
	if err != nil {
		return nil, err
	}
	cp := &checkpoint.Checkpoint{FormatVersion: 2, Config: raw, Tensors: make(map[string]checkpoint.Tensor, len(m.tensors))}
	for name, t := range m.tensors {
		switch {
		case strings.HasPrefix(name, "stack/"):
			if strings.HasPrefix(name, "stack/final_norm/") {
				cp.Tensors[name] = cloneCheckpointTensor(t)
				continue
			}
			if len(t.Shape) == 0 || t.Shape[0] != m.config.Layers {
				return nil, fmt.Errorf("needle: unsupported stacked tensor geometry %s %v", name, t.Shape)
			}
			sliced, err := tensorSliceAxis(t, 0, selected)
			if err != nil {
				return nil, fmt.Errorf("needle: slice %s: %w", name, err)
			}
			cp.Tensors[name] = sliced
		case strings.HasPrefix(name, "embedding_head/"), strings.HasPrefix(name, "confidence_head/"), strings.HasPrefix(name, "router_head/"):
			suffix := name[strings.IndexByte(name, '/')+1:]
			switch suffix {
			case "probes", "gain":
				if len(t.Shape) == 0 || t.Shape[0] != m.config.Layers+1 {
					return nil, fmt.Errorf("needle: unsupported head tensor geometry %s %v", name, t.Shape)
				}
				sliced, err := tensorSliceAxis(t, 0, rows)
				if err != nil {
					return nil, fmt.Errorf("needle: slice %s: %w", name, err)
				}
				cp.Tensors[name] = sliced
			case "row_bias":
				if len(t.Shape) < 2 || t.Shape[1] != m.config.Layers+1 {
					return nil, fmt.Errorf("needle: unsupported head tensor geometry %s %v", name, t.Shape)
				}
				sliced, err := tensorSliceAxis(t, 1, rows)
				if err != nil {
					return nil, fmt.Errorf("needle: slice %s: %w", name, err)
				}
				cp.Tensors[name] = sliced
			default:
				cp.Tensors[name] = cloneCheckpointTensor(t)
			}
		case strings.HasPrefix(name, "engrams_"):
			site, rest, ok := parseEngramTensorName(name)
			if !ok {
				return nil, fmt.Errorf("needle: unsupported engram tensor name %s", name)
			}
			if site >= len(m.config.EngramLayers) {
				return nil, fmt.Errorf("needle: engram tensor %s site %d out of range", name, site)
			}
			if !selectedSet[m.config.EngramLayers[site]] {
				continue
			}
			newSite := 0
			for _, layer := range m.config.EngramLayers[:site] {
				if selectedSet[layer] {
					newSite++
				}
			}
			cp.Tensors[fmt.Sprintf("engrams_%d%s", newSite, rest)] = cloneCheckpointTensor(t)
		default:
			cp.Tensors[name] = cloneCheckpointTensor(t)
		}
	}
	child, err := New(cp)
	if err != nil {
		return nil, err
	}
	if len(m.packed) > 0 {
		child.packed = make(map[packedKey]*simd.CQMatrix)
		for key, p := range m.packed {
			next := key
			if key.layer >= 0 {
				idx := slices.Index(selected, key.layer)
				if idx < 0 {
					continue
				}
				next.layer = idx
			}
			if strings.HasPrefix(key.name, "engrams_") {
				site, rest, ok := parseEngramTensorName(key.name)
				if !ok || !selectedSet[m.config.EngramLayers[site]] {
					continue
				}
				newSite := 0
				for _, layer := range m.config.EngramLayers[:site] {
					if selectedSet[layer] {
						newSite++
					}
				}
				next.name = fmt.Sprintf("engrams_%d%s", newSite, rest)
			}
			child.packed[next] = p // immutable matrices can be shared
		}
	}
	return child, nil
}
