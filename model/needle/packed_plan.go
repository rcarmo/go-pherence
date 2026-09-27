package needle

import (
	"fmt"
	"slices"
	"sort"
	"strings"
)

// packedOnlyPlan is an inventory, not permission to discard decoded storage.
// The loader, forward path, decoder preparation, slicing and checkpoint export
// still use F32 tensors. Counts are logical tensor bytes, not RSS or savings.
type packedOnlyPlan struct {
	PotentiallyReplaceable []packedPlanTensor
	RequiredDecoded        []packedPlanTensor
	Blockers               []string
}

type packedPlanTensor struct {
	Name  string
	Bytes int64
	Why   string
}

// planPackedOnly inspects an already loaded, decoded archive without modifying
// it. Only whole tensors whose every executed matrix site has a valid packed
// counterpart are coverage candidates. Everything else stays decoded.
func planPackedOnly(m *Model) (packedOnlyPlan, error) {
	var plan packedOnlyPlan
	if m == nil || !m.deployed {
		return plan, fmt.Errorf("needle: packed-only planning requires a decoded deployed archive")
	}
	shapes := expectedShapes(m.config)
	for name, want := range shapes {
		got, ok := m.tensors[name]
		if !ok || !slices.Equal(got.Shape, want) {
			return packedOnlyPlan{}, fmt.Errorf("needle: packed-only plan missing/invalid tensor %s", name)
		}
	}
	// Validate every key, including orphans and partial tensors. In particular
	// don't count a packed matrix with a plausible size under an unused name.
	for key, p := range m.packed {
		t, ok := m.tensors[key.name]
		if !ok || p == nil {
			return packedOnlyPlan{}, fmt.Errorf("needle: packed-only plan invalid key %+v", key)
		}
		rows, cols, layers, ok := packedPlanGeometry(m.config, key.name, t.Shape)
		layered := len(t.Shape) == 3 && key.name != "embedding/embedding"
		if !ok || (!layered && key.layer != -1) || (layered && (key.layer < 0 || key.layer >= layers)) || p.Rows() != rows || p.Cols() != cols {
			return packedOnlyPlan{}, fmt.Errorf("needle: packed-only plan invalid geometry/key %+v", key)
		}
	}
	names := make([]string, 0, len(m.tensors))
	for name := range m.tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		t := m.tensors[name]
		entry := packedPlanTensor{Name: name, Bytes: int64(len(t.Data)) * 4}
		_, _, layers, eligible := packedPlanGeometry(m.config, name, t.Shape)
		if isPackedPlanHead(name) {
			entry.Why = "packed archive head projection is not consumed: headWeight/probeHead use decoded F32"
			plan.RequiredDecoded = append(plan.RequiredDecoded, entry)
			continue
		}
		if !eligible {
			entry.Why = "no packed execution consumer (norm, taps, gather, or auxiliary/export data)"
			plan.RequiredDecoded = append(plan.RequiredDecoded, entry)
			continue
		}
		missing := 0
		for layer := 0; layer < layers; layer++ {
			keyLayer := layer
			if len(t.Shape) == 2 {
				keyLayer = -1
			}
			if m.packed[packedKey{name, keyLayer}] == nil {
				missing++
			}
		}
		if missing > 0 {
			entry.Why = fmt.Sprintf("missing %d of %d packed execution sites; partial tensors stay decoded", missing, layers)
			plan.RequiredDecoded = append(plan.RequiredDecoded, entry)
			continue
		}
		entry.Why = "all packed execution sites covered; runtime/storage migration still required"
		plan.PotentiallyReplaceable = append(plan.PotentiallyReplaceable, entry)
	}
	plan.Blockers = []string{
		"archive loading materializes and validates F32 before building CQ matrices",
		"newModel requires F32 data for every expected tensor; shapes have no independent owner",
		"trunk token embedding gather still reads decoded embedding (even with packed output projection)",
		"execution.param and NewDecoder prepare decoded views for every expected tensor; probeHead also reads head weights as F32",
		"SliceDepth and Checkpoint clone/export decoded tensors; packed-only policy is undefined",
	}
	return plan, nil
}

// packedPlanGeometry lists only keys actually consumed by e.linear or the
// tied embedding/output path. Internal storage is transposed for linear weights.
func isPackedPlanHead(name string) bool {
	switch name {
	case "embedding_head/proj/kernel", "confidence_head/proj/kernel", "router_head/proj/kernel", "contrastive_head/proj/kernel":
		return true
	}
	return false
}

func packedPlanGeometry(c Config, name string, shape []int) (rows, cols, layers int, ok bool) {
	if name == "embedding/embedding" && len(shape) == 2 {
		return shape[0], shape[1], 1, true
	}
	// Archive heads can retain CQ for transposed projection records, but
	// probeHead calls headWeight/param rather than e.linear. Validate those
	// payloads without counting them as executable packed coverage.
	if isPackedPlanHead(name) && len(shape) == 2 {
		return shape[1], shape[0], 1, true
	}
	linear := false
	if strings.HasPrefix(name, "stack/layers/block/self_attn/") {
		switch strings.TrimPrefix(name, "stack/layers/block/self_attn/") {
		case "q_proj/kernel", "k_proj/kernel", "v_proj/kernel", "gate_proj/kernel", "out_proj/kernel":
			linear = true
		}
	} else {
		switch name {
		case "stack/mhc_phi_pre", "stack/mhc_phi_post", "stack/mhc_phi_res":
			linear = true
		}
		for i := range c.EngramLayers {
			if name == fmt.Sprintf("engrams_%d/key_proj/kernel", i) || name == fmt.Sprintf("engrams_%d/value_proj/kernel", i) {
				linear = true
			}
		}
	}
	if !linear {
		return 0, 0, 0, false
	}
	if strings.HasPrefix(name, "engrams_") && len(shape) == 2 {
		return shape[1], shape[0], 1, true
	}
	if len(shape) != 3 || shape[0] != c.Layers {
		return 0, 0, 0, false
	}
	return shape[2], shape[1], c.Layers, true
}
