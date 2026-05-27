package speaker

import "testing"

func TestSmoothSingletonLabelsMergesSimilarSingletons(t *testing.T) {
	labels := []int{0, 0, 1, 0, 2}
	embeddings := [][]float32{
		{1, 0},
		{0.99, 0.01},
		{0.98, 0.02},
		{0.97, 0.03},
		{0.96, 0.04},
	}
	smoothed := SmoothSingletonLabels(labels, embeddings, 0.4)
	for i, label := range smoothed {
		if label != 0 {
			t.Fatalf("label[%d]=%d want all merged to 0: %v", i, label, smoothed)
		}
	}
}

func TestSmoothSingletonLabelsKeepsDissimilarSingleton(t *testing.T) {
	labels := []int{0, 0, 1}
	embeddings := [][]float32{{1, 0}, {0.9, 0.1}, {0, 1}}
	smoothed := SmoothSingletonLabels(labels, embeddings, 0.4)
	if len(smoothed) != 3 || smoothed[2] != 1 {
		t.Fatalf("smoothed=%v, want dissimilar singleton preserved", smoothed)
	}
}

func TestRenumberLabels(t *testing.T) {
	got := RenumberLabels([]int{7, 7, 3, 9, 3})
	want := []int{0, 0, 1, 2, 1}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("RenumberLabels=%v want %v", got, want)
		}
	}
}
