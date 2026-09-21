package needle

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

func widthPrefixIndices(n int) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	return idx
}

func tensorSlicePrefix(t checkpoint.Tensor, axis, n int) (checkpoint.Tensor, error) {
	return tensorSliceAxis(t, axis, widthPrefixIndices(n))
}

func tensorSliceLast(t checkpoint.Tensor, n int) (checkpoint.Tensor, error) {
	if len(t.Shape) == 0 {
		return checkpoint.Tensor{}, fmt.Errorf("rank-0 tensor")
	}
	return tensorSlicePrefix(t, len(t.Shape)-1, n)
}

func tensorSliceHeadProjInput(t checkpoint.Tensor, parentWidth, childWidth int) (checkpoint.Tensor, error) {
	if len(t.Shape) != 2 || parentWidth < 1 || childWidth < 1 || childWidth > parentWidth || t.Shape[0]%parentWidth != 0 {
		return checkpoint.Tensor{}, fmt.Errorf("invalid head projection shape %v for width %d", t.Shape, parentWidth)
	}
	groups, out := t.Shape[0]/parentWidth, t.Shape[1]
	next := checkpoint.Tensor{Shape: []int{groups * childWidth, out}, Data: make([]float32, groups*childWidth*out)}
	for g := 0; g < groups; g++ {
		src := g * parentWidth * out
		dst := g * childWidth * out
		copy(next.Data[dst:dst+childWidth*out], t.Data[src:src+childWidth*out])
	}
	return next, nil
}

func (m *Model) slicedWidthRawConfig(width, heads, seedHeads int) (json.RawMessage, Config, error) {
	fields := map[string]json.RawMessage{}
	if len(m.rawConfig) > 0 {
		if err := json.Unmarshal(m.rawConfig, &fields); err != nil {
			return nil, Config{}, fmt.Errorf("needle: decode raw config: %w", err)
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
	for _, kv := range []struct {
		key string
		val any
	}{
		{"d_model", width},
		{"num_heads", heads},
		{"engram_seed_heads", seedHeads},
		{"ladder_depths", []int{}},
		{"ladder_sample", false},
		{"ladder_widths", []int{}},
	} {
		if err := set(kv.key, kv.val); err != nil {
			return nil, Config{}, err
		}
	}
	out, err := json.Marshal(fields)
	if err != nil {
		return nil, Config{}, err
	}
	var child Config
	if err := json.Unmarshal(out, &child); err != nil {
		return nil, Config{}, err
	}
	if child.Generation == 0 {
		if child.QKDim > 0 {
			child.Generation = 3
		} else if child.AttnDim > 0 {
			child.Generation = 2
		} else {
			return nil, Config{}, fmt.Errorf("needle: ambiguous generation; specify generation in checkpoint config")
		}
	}
	if err := child.validate(); err != nil {
		return nil, Config{}, err
	}
	return out, child, nil
}

// SliceWidth returns a reduced-width Needle3 rung using upstream's exact half-width tensor cuts.
func (m *Model) SliceWidth(width int) (*Model, error) {
	if m == nil {
		return nil, fmt.Errorf("needle: nil model")
	}
	if m.config.Generation != 3 {
		return nil, fmt.Errorf("needle: SliceWidth is only supported for Needle3 checkpoints")
	}
	if m.deployed {
		return nil, fmt.Errorf("needle: SliceWidth requires source model weights")
	}
	parentWidth := m.config.DModel
	if width < 1 || width != parentWidth/2 || parentWidth%2 != 0 {
		return nil, fmt.Errorf("needle: SliceWidth only supports half-width slicing from %d to %d (got %d)", parentWidth, parentWidth/2, width)
	}
	if parentWidth&(parentWidth-1) != 0 {
		return nil, fmt.Errorf("needle: SliceWidth requires a power-of-two parent width")
	}
	if !slices.Contains(m.config.LadderWidths, width) {
		return nil, fmt.Errorf("needle: SliceWidth requires the requested rung in trained ladder_widths")
	}
	if m.config.Heads%2 != 0 {
		return nil, fmt.Errorf("needle: SliceWidth requires an even num_heads, got %d", m.config.Heads)
	}
	childHeads := m.config.Heads / 2
	if childHeads < m.config.KVHeads || childHeads%m.config.KVHeads != 0 {
		return nil, fmt.Errorf("needle: SliceWidth would produce num_heads=%d incompatible with num_kv_heads=%d", childHeads, m.config.KVHeads)
	}
	_, parentBB := hadaBlocks(parentWidth)
	baChild, childBB := hadaBlocks(width)
	if childBB != parentBB {
		return nil, fmt.Errorf("needle: SliceWidth width %d is incompatible with parent Hadamard geometry (%d != %d)", width, childBB, parentBB)
	}
	for name := range m.tensors {
		if strings.HasPrefix(name, "ab_scales/") {
			return nil, fmt.Errorf("needle: SliceWidth does not support AB scales")
		}
	}
	for _, kind := range []HeadKind{Embedding, Confidence, Router} {
		if _, ok := m.tensors[string(kind)+"_head/probes"]; ok {
			if _, _, _, err := m.headGeometry(kind); err != nil {
				return nil, err
			}
		}
	}

	var (
		orders     []int
		engHeads   int
		subDim     int
		keptTables []int
		kvRows     []int
	)
	if len(m.config.EngramLayers) > 0 {
		orders = append([]int(nil), m.config.EngramOrders...)
		engHeads = m.config.EngramHeads
		if engHeads%2 != 0 {
			return nil, fmt.Errorf("needle: SliceWidth requires an even engram head count, got %d", engHeads)
		}
		subDim = parentWidth / (len(orders) * engHeads)
		keptTables = make([]int, 0, len(orders)*engHeads/2)
		for oi := range orders {
			base := oi * engHeads
			for h := 0; h < engHeads/2; h++ {
				keptTables = append(keptTables, base+h)
			}
		}
		kvRows = make([]int, 0, len(keptTables)*subDim)
		for _, table := range keptTables {
			base := table * subDim
			for i := 0; i < subDim; i++ {
				kvRows = append(kvRows, base+i)
			}
		}
	}
	seedHeads := m.config.EngramSeedHeads
	if seedHeads == 0 && len(m.config.EngramLayers) > 0 {
		seedHeads = engHeads
	}
	raw, childConfig, err := m.slicedWidthRawConfig(width, childHeads, seedHeads)
	if err != nil {
		return nil, err
	}
	if len(m.config.EngramLayers) > 0 {
		wantHeads := engHeads / 2
		wantTables := len(orders) * wantHeads
		cfgTables := len(childConfig.EngramOrders) * childConfig.EngramHeads
		cfgSub := childConfig.DModel / cfgTables
		if childConfig.EngramHeads != wantHeads || cfgTables != wantTables || cfgSub != subDim {
			if m.config.EngramHeads != 0 {
				return nil, fmt.Errorf("needle: SliceWidth cannot preserve explicit engram_heads=%d: child width %d needs %d heads and sub-dimension %d to match sliced tensors", m.config.EngramHeads, width, wantHeads, subDim)
			}
			return nil, fmt.Errorf("needle: SliceWidth child engram geometry mismatch: config wants %d tables of width %d, sliced tensors provide %d tables of width %d", cfgTables, cfgSub, wantTables, subDim)
		}
	}

	qCols := childHeads * m.config.QKDim
	oRows := childHeads * m.config.VDim
	laneRows := make([]int, 0, m.config.Lanes*width)
	for lane := 0; lane < m.config.Lanes; lane++ {
		base := lane * parentWidth
		for i := 0; i < width; i++ {
			laneRows = append(laneRows, base+i)
		}
	}

	cp := m.checkpointView()
	cp.Config = raw
	for name, t := range m.tensors {
		var next checkpoint.Tensor
		switch {
		case strings.HasPrefix(name, "engrams_"):
			switch {
			case strings.HasSuffix(name, "/embedding"):
				next, err = tensorSliceAxis(t, 0, keptTables)
			case strings.HasSuffix(name, "key_proj/kernel"), strings.HasSuffix(name, "value_proj/kernel"):
				next, err = tensorSliceAxis(t, 0, kvRows)
				if err == nil {
					next, err = tensorSlicePrefix(next, 1, width)
				}
			case strings.HasSuffix(name, "/taps"):
				next, err = tensorSliceLast(t, width)
			default:
				continue
			}
		case strings.HasPrefix(name, "embedding_head/"), strings.HasPrefix(name, "confidence_head/"), strings.HasPrefix(name, "router_head/"):
			switch {
			case strings.HasSuffix(name, "/probes"), strings.HasSuffix(name, "/query"):
				next, err = tensorSliceLast(t, width)
			case strings.HasSuffix(name, "proj/kernel"):
				next, err = tensorSliceHeadProjInput(t, parentWidth, width)
			default:
				continue
			}
		case strings.Contains(name, "mhc_phi"):
			next, err = tensorSliceAxis(t, 1, laneRows)
		case strings.Contains(name, "hadamard_mlp/"):
			last := name[strings.LastIndexByte(name, '/')+1:]
			switch last {
			case "w1a", "w2a", "w3a":
				next, err = tensorSlicePrefix(t, 1, baChild)
				if err == nil {
					next, err = tensorSlicePrefix(next, 2, baChild)
				}
			case "d1", "d2", "b2", "d3", "d4":
				next, err = tensorSliceLast(t, width)
			case "cond_v":
				next, err = tensorSlicePrefix(t, 1, width)
			case "cond_u":
				next, err = tensorSliceLast(t, width)
			default:
				continue
			}
		case strings.HasSuffix(name, "q_proj/kernel"):
			next, err = tensorSlicePrefix(t, 1, width)
			if err == nil {
				next, err = tensorSlicePrefix(next, 2, qCols)
			}
		case strings.HasSuffix(name, "gate_proj/kernel"):
			next, err = tensorSlicePrefix(t, 1, width)
			if err == nil {
				next, err = tensorSlicePrefix(next, 2, oRows)
			}
		case strings.HasSuffix(name, "k_proj/kernel"), strings.HasSuffix(name, "v_proj/kernel"):
			next, err = tensorSlicePrefix(t, 1, width)
		case strings.HasSuffix(name, "out_proj/kernel"):
			next, err = tensorSlicePrefix(t, 1, oRows)
			if err == nil {
				next, err = tensorSlicePrefix(next, 2, width)
			}
		case strings.HasSuffix(name, "/q_taps"):
			next, err = tensorSliceLast(t, qCols)
		case strings.HasSuffix(name, "/k_taps"), strings.HasSuffix(name, "/v_taps"):
			continue
		case strings.HasSuffix(name, "q_norm/scale"), strings.HasSuffix(name, "k_norm/scale"):
			continue
		case strings.HasSuffix(name, "/scale"):
			next, err = tensorSliceLast(t, width)
		case name == "embedding/embedding":
			next, err = tensorSliceLast(t, width)
		default:
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("needle: slice %s: %w", name, err)
		}
		cp.Tensors[name] = next
	}
	child, err := newModel(cp, true)
	if err != nil {
		return nil, err
	}
	for _, kind := range []HeadKind{Embedding, Confidence, Router} {
		if _, ok := child.tensors[string(kind)+"_head/probes"]; ok {
			if _, _, _, err := child.headGeometry(kind); err != nil {
				return nil, err
			}
		}
	}
	return child, nil
}
