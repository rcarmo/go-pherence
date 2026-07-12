package diffusiongemma

import (
	"math"
	"testing"
)

func TestPoolVisionGridToSoftTokensF32UsesSpatialKernels(t *testing.T) {
	// A 4x2 grid pooled with k=2 must group left/right 2x2 blocks, not
	// contiguous rows in flattened patch order.
	patchHidden := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	pooled, err := PoolVisionGridToSoftTokensF32(patchHidden, 4, 2, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{3.5, 5.5}
	for i := range want {
		if pooled[i] != want[i] {
			t.Fatalf("pooled[%d]=%v want %v full=%v", i, pooled[i], want[i], pooled)
		}
	}
}

func TestPoolVisionGridToSoftTokensF32RejectsNonSquareKernel(t *testing.T) {
	if _, err := PoolVisionGridToSoftTokensF32(make([]float32, 6), 3, 2, 1, 2); err == nil {
		t.Fatal("expected non-square kernel rejection")
	}
}

func TestPoolVisionPatchesToSoftTokensF32(t *testing.T) {
	patchHidden := []float32{
		1, 3,
		5, 7,
		2, 4,
		6, 8,
	}
	pooled, err := PoolVisionPatchesToSoftTokensF32(patchHidden, 4, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{3, 5, 4, 6}
	for i := range want {
		if pooled[i] != want[i] {
			t.Fatalf("pooled[%d]=%v want %v full=%v", i, pooled[i], want[i], pooled)
		}
	}
}

func TestPoolVisionPatchesToSoftTokensF32RequiresEvenGroups(t *testing.T) {
	if _, err := PoolVisionPatchesToSoftTokensF32(make([]float32, 6), 3, 2, 2); err == nil {
		t.Fatal("expected uneven patch/soft-token grouping error")
	}
}

func TestRunVisionTowerAndPoolF32Synthetic(t *testing.T) {
	layers := []VisionLayerF32{tinyVisionLayerF32(2, 1, 2, 3)}
	patchHidden := []float32{1, -0.5, 0.25, 0.75}
	pooled, err := RunVisionTowerAndPoolF32(patchHidden, 2, 2, 1, 2, 1, layers)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1.813806, 0.651149}
	for i := range want {
		if math.Abs(float64(pooled[i]-want[i])) > 1e-5 {
			t.Fatalf("pooled[%d]=%.6f want %.6f full=%v", i, pooled[i], want[i], pooled)
		}
	}
}
