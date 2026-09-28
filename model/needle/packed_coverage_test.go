package needle

import (
	"context"
	"strings"
	"testing"
)

// TestPackedArchiveCoverage inventories the hybrid fixture before any attempt
// to omit decoded CQ tensors. It asserts geometry/ownership and a live packed
// decoder session, but it does not claim packed-only inference or lower RSS.
func TestPackedArchiveCoverage(t *testing.T) {
	m, _, err := LoadArchive("../../loader/needle/testdata/needle3.cact")
	if err != nil {
		t.Fatal(err)
	}
	if !m.deployed || len(m.packed) == 0 || len(m.tensors) == 0 {
		t.Fatal("fixture is not the hybrid deployed archive")
	}
	var decodedBytes, replaceableBytes, plainBytes, packedBytes int64
	seen := make(map[string]int)
	for name, tensor := range m.tensors {
		n := int64(len(tensor.Data)) * 4
		decodedBytes += n
		if strings.HasPrefix(name, "stack/layers/block/") && len(tensor.Shape) > 0 && tensor.Shape[0] == m.config.Layers {
			for layer := 0; layer < m.config.Layers; layer++ {
				key := packedKey{name, layer}
				if packed := m.packed[key]; packed != nil {
					if len(tensor.Shape) != 3 || packed.Cols() != tensor.Shape[1] || packed.Rows() != tensor.Shape[2] {
						t.Fatalf("packed geometry %s layer %d shape=%v cols/rows=%d/%d", name, layer, tensor.Shape, packed.Cols(), packed.Rows())
					}
					seen[name]++
					replaceableBytes += n / int64(m.config.Layers)
				}
			}
		} else if packed := m.packed[packedKey{name, -1}]; packed != nil {
			rows, cols := 0, 0
			if len(tensor.Shape) == 2 {
				rows, cols = tensor.Shape[1], tensor.Shape[0] // transposed projections
				if name == "embedding/embedding" {
					rows, cols = tensor.Shape[0], tensor.Shape[1]
				}
			}
			if packed.Rows() != rows || packed.Cols() != cols {
				t.Fatalf("packed geometry %s shape=%v cols/rows=%d/%d", name, tensor.Shape, packed.Cols(), packed.Rows())
			}
			seen[name]++
			replaceableBytes += n
		}
	}
	plainBytes = decodedBytes - replaceableBytes
	for key, p := range m.packed {
		if p == nil || seen[key.name] == 0 {
			t.Fatalf("packed key without decoded counterpart: %+v", key)
		}
		packedBytes += p.Bytes()
	}
	if replaceableBytes <= 0 || plainBytes <= 0 || packedBytes <= 0 || packedBytes != m.PackedBytes() {
		t.Fatalf("invalid coverage decoded=%d replaceable=%d plain=%d packed=%d", decodedBytes, replaceableBytes, plainBytes, packedBytes)
	}
	if m.packed[packedKey{"embedding/embedding", -1}] == nil {
		t.Fatal("fixture lacks packed tied embedding/output")
	}
	// The current trunk gathers token embeddings from decoded storage even in
	// packed mode, and NewDecoder prepares views for every expected tensor.
	// Omission must address both paths before calling the archive packed-only.
	decoder, err := m.NewDecoder(DecoderOptions{Capacity: 12, Execution: Options{Packed: true}})
	if err != nil {
		t.Fatal(err)
	}
	out, err := decoder.Step(context.Background(), 2)
	if err != nil || len(out) != m.config.OutVocab {
		t.Fatalf("packed decoder step logits=%d err=%v", len(out), err)
	}
	t.Logf("tiny hybrid archive: decoded=%d bytes potentially CQ-covered=%d bytes required dense=%d bytes packed additional=%d bytes matrices=%d tensor names=%d (not peak RSS)", decodedBytes, replaceableBytes, plainBytes, packedBytes, len(m.packed), len(m.tensors))
}
