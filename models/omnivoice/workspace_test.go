package omnivoice

import (
	"math"
	"testing"
)

func TestWorkspaceParityAndReuse(t *testing.T) {
	b, f := loadFixture(t)
	s, err := b.NewWorkspace(3)
	if err != nil {
		t.Fatal(err)
	}
	dst := make([]float32, len(f.Input.Data))
	for i := 0; i < 5; i++ {
		name := "unmasked"
		var mask []float32
		if i%2 == 1 {
			name = "mask_block_key2"
			mask = f.Mask.Data
		}
		if err = b.ForwardInto(dst, f.Input.Data, 3, nil, mask, s); err != nil {
			t.Fatal(err)
		}
		for j, v := range dst {
			if diff := math.Abs(float64(v - f.Outputs[name].Data[j])); diff > 2e-5 || math.IsNaN(diff) {
				t.Fatalf("run %d index %d diff %g", i, j, diff)
			}
		}
	}
	// Exact input/output alias is supported for layer chaining.
	copy(dst, f.Input.Data)
	if err = b.ForwardInto(dst, dst, 3, nil, nil, s); err != nil {
		t.Fatal(err)
	}
	for j, v := range dst {
		if math.Abs(float64(v-f.Outputs["unmasked"].Data[j])) > 2e-5 {
			t.Fatal("alias parity")
		}
	}
}
func TestWorkspaceZeroAllocs(t *testing.T) {
	b, f := loadFixture(t)
	s, _ := b.NewWorkspace(3)
	dst := make([]float32, len(f.Input.Data))
	var err error
	n := testing.AllocsPerRun(100, func() { err = b.ForwardInto(dst, f.Input.Data, 3, nil, nil, s) })
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Fatalf("allocations=%g", n)
	}
}
func TestWorkspacePositionsAndValidation(t *testing.T) {
	b, f := loadFixture(t)
	s, _ := b.NewWorkspace(3)
	dst := make([]float32, len(f.Input.Data))
	for _, positions := range [][]int{{3, 8, 12}, nil, {10, 20, 30}} {
		want, err := b.Forward(f.Input.Data, 3, positions, nil)
		if err != nil {
			t.Fatal(err)
		}
		if err = b.ForwardInto(dst, f.Input.Data, 3, positions, nil, s); err != nil {
			t.Fatal(err)
		}
		for i, v := range dst {
			if v != want[i] {
				t.Fatal("position cache invalidation")
			}
		}
	}
	if err := b.ForwardInto(dst, f.Input.Data, 3, nil, nil, nil); err == nil {
		t.Fatal("nil workspace accepted")
	}
	if _, err := b.NewWorkspace(-1); err == nil {
		t.Fatal("negative capacity accepted")
	}
	if _, err := b.NewWorkspace(int(^uint(0) >> 1)); err == nil {
		t.Fatal("overflow accepted")
	}
}
func BenchmarkTinyBlockInto(b *testing.B) {
	block, f := loadFixture(b)
	s, _ := block.NewWorkspace(3)
	dst := make([]float32, len(f.Input.Data))
	b.ReportAllocs()
	for b.Loop() {
		if err := block.ForwardInto(dst, f.Input.Data, 3, nil, nil, s); err != nil {
			b.Fatal(err)
		}
	}
}
