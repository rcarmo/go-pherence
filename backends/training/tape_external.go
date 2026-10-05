package training

import "fmt"

// ValidateExternalTapeBindings requires every shape-only external source to be
// explicitly bound before execution, including unused sources before pruning.
// Backend-specific extent/liveness/alias admission must follow. Host constants
// may also be rebound read-only; writable outputs may never be bound.
func ValidateExternalTapeBindings[B any](t Tape, bindings map[int]B) error {
	for id, s := range t.Tensors {
		if s.External {
			if _, ok := bindings[id]; !ok {
				return fmt.Errorf("external tensor%d missing binding", id)
			}
		}
	}
	for id := range bindings {
		if id < 0 || id >= len(t.Tensors) || (!t.Tensors[id].External && t.Tensors[id].Values == nil) {
			return fmt.Errorf("bound tensor is not read-only source")
		}
	}
	return nil
}

// WithExternalTapeInputs owns graph metadata and removes only explicitly named
// host constants. This is a shape/provenance declaration, not numerical approval:
// callers must bind the real device buffers and keep original runtime guards.
// Complete source admission occurs before conversion and after it; it cannot
// hide nonfinite dead constants or permit rebinding computed destinations.
func WithExternalTapeInputs(t Tape, ids []int, maxBytes int64) (Tape, error) {
	if err := t.Validate(maxBytes); err != nil {
		return Tape{}, err
	}
	if len(ids) == 0 || len(ids) > 4096 {
		return Tape{}, fmt.Errorf("external input count")
	}
	seen := map[int]bool{}
	for _, id := range ids {
		if id < 0 || id >= len(t.Tensors) || t.Tensors[id].Values == nil || t.Tensors[id].External || seen[id] {
			return Tape{}, fmt.Errorf("external input ID/constant/duplicate")
		}
		seen[id] = true
	}
	// Removed constants have already been fully validated and are never used
	// as host inputs again. Own every remaining value, without copying then
	// discarding the selected (potentially model-sized) arrays.
	out := copyTapeStorage(t, seen, true)
	for _, id := range ids {
		out.Tensors[id].Values = nil
		out.Tensors[id].External = true
	}
	if err := out.Validate(maxBytes); err != nil {
		return Tape{}, err
	}
	return out, nil
}
