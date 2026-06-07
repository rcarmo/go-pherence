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
		t.Error("should find attn marker (case-insensitive substring)")
	}
	if !AnyTensorMarker(names, []string{"EMBED"}) {
		t.Error("match should be case-insensitive")
	}
	if AnyTensorMarker(names, []string{"router"}) {
		t.Error("absent marker should not match")
	}
}
