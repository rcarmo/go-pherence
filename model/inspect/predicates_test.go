package inspect

import "testing"

func TestMatrixMatches(t *testing.T) {
	if !MatrixMatches([]int{4, 8}, 4, 8) {
		t.Error("exact orientation should match")
	}
	if !MatrixMatches([]int{8, 4}, 4, 8) {
		t.Error("transposed orientation should match")
	}
	if MatrixMatches([]int{4, 8, 1}, 4, 8) {
		t.Error("non-2D should not match")
	}
	if MatrixMatches([]int{4, 9}, 4, 8) {
		t.Error("wrong cols should not match")
	}
}

func TestIsPlaceholder(t *testing.T) {
	if !IsPlaceholder("pending-foo") {
		t.Error("pending- prefix is a placeholder")
	}
	if IsPlaceholder("ready") {
		t.Error("non-pending is not a placeholder")
	}
}

func TestAnyTensorMarker(t *testing.T) {
	names := []string{"model.layers.0.attn", "model.embed"}
	if !AnyTensorMarker(names, []string{"attn"}) {
		t.Error("should find attn marker (substring of a lowercased name)")
	}
	if !AnyTensorMarker([]string{"model.EMBED"}, []string{"embed"}) {
		t.Error("name is lowercased before matching, so mixed-case names match lowercase markers")
	}
	if AnyTensorMarker(names, []string{"router"}) {
		t.Error("absent marker should not match")
	}
}
