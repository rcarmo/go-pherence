package needle

import (
	"fmt"
	"slices"

	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

// CompactPacked returns a separate inference-only, opt-in model view. It
// retains dense-required tensors and immutable CQ matrices while dropping
// decoded data for whole tensors covered by packed execution sites. Loading
// the archive still materialises F32 first; this method does not bound or
// reduce peak load-time RSS. The original model remains unchanged.
func (m *Model) CompactPacked() (*Model, error) {
	plan, err := planPackedOnly(m)
	if err != nil {
		return nil, err
	}
	if len(plan.PotentiallyReplaceable) == 0 || m.compactPacked {
		return nil, fmt.Errorf("needle: no eligible decoded tensors in deployed archive")
	}
	if m.config.Generation != 3 {
		return nil, fmt.Errorf("needle: compact packed inference is limited to Needle3")
	}
	compact := *m
	compact.tensors = make(map[string]checkpoint.Tensor, len(plan.RequiredDecoded))
	for _, entry := range plan.RequiredDecoded {
		compact.tensors[entry.Name] = m.tensors[entry.Name]
	}
	compact.packedShapes = make(map[string][]int, len(plan.PotentiallyReplaceable))
	for _, entry := range plan.PotentiallyReplaceable {
		compact.packedShapes[entry.Name] = slices.Clone(m.tensors[entry.Name].Shape)
	}
	compact.compactPacked = true
	return &compact, nil
}

// DecodedBytes counts currently retained decoded F32 tensor data. The packed
// matrix payloads are additional and reported separately by PackedBytes.
func (m *Model) DecodedBytes() int64 {
	if m == nil {
		return 0
	}
	var n int64
	for _, tensor := range m.tensors {
		n += int64(len(tensor.Data)) * 4
	}
	return n
}

func (m *Model) parameterShape(name string) []int {
	if tensor, ok := m.tensors[name]; ok {
		return tensor.Shape
	}
	return m.packedShapes[name]
}
