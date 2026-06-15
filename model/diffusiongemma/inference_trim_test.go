package diffusiongemma

import "testing"

func TestTrimCanvasLikeLlamaCutsAtEOG(t *testing.T) {
	cut, reason := trimCanvasLikeLlama([]int{100, 101, 1, 102}, map[int]bool{1: true})
	if cut != 2 || reason != "eog" {
		t.Fatalf("cut=%d reason=%q want 2/eog", cut, reason)
	}
}

func TestTrimCanvasLikeLlamaCutsAtRepetitionStrideOne(t *testing.T) {
	cut, reason := trimCanvasLikeLlama([]int{100, 7, 7, 7, 7, 7, 7, 7, 101}, nil)
	if cut != 1 || reason != "repetition" {
		t.Fatalf("cut=%d reason=%q want 1/repetition", cut, reason)
	}
}

func TestTrimCanvasLikeLlamaCutsAtRepetitionStrideTwo(t *testing.T) {
	cut, reason := trimCanvasLikeLlama([]int{100, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 101}, nil)
	if cut != 1 || reason != "repetition" {
		t.Fatalf("cut=%d reason=%q want 1/repetition", cut, reason)
	}
}

func TestTrimCanvasLikeLlamaLeavesUntrimmedCanvas(t *testing.T) {
	canvas := []int{100, 101, 102}
	cut, reason := trimCanvasLikeLlama(canvas, map[int]bool{1: true})
	if cut != len(canvas) || reason != "" {
		t.Fatalf("cut=%d reason=%q want %d/empty", cut, reason, len(canvas))
	}
}
