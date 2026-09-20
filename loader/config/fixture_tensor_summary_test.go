package config

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"testing"
)

func TestSummarizeFixtureFloat32Tensor(t *testing.T) {
	got, err := SummarizeFixtureFloat32Tensor("latent", []int{2, 2}, []float32{1, -1, 3, 5})
	if err != nil {
		t.Fatal(err)
	}
	if got.Name != "latent" || got.DType != "float32" || got.Min != -1 || got.Max != 5 || got.Mean != 2 {
		t.Fatalf("summary=%+v", got)
	}
	if got.SHA256LEF32 != "90f73a7925d83a9ec35185cd7814055c7ad4e82c6cd791d53744d130781ab610" {
		t.Fatalf("hash=%s", got.SHA256LEF32)
	}
	if len(got.Shape) != 2 || got.Shape[0] != 2 || len(got.FirstValues) != 4 || got.FirstValues[2] != 3 {
		t.Fatalf("summary shape/first values=%+v", got)
	}
	got.Shape[0] = 99
	got.FirstValues[0] = 99
	again, err := SummarizeFixtureFloat32Tensor("latent", []int{2, 2}, []float32{1, -1, 3, 5})
	if err != nil {
		t.Fatal(err)
	}
	if again.Shape[0] != 2 || again.FirstValues[0] != 1 {
		t.Fatalf("summary aliases caller-visible state: %+v", again)
	}
}

func TestSummarizeFixtureFloat32TensorRejectsBadShapes(t *testing.T) {
	if _, err := SummarizeFixtureFloat32Tensor("bad", []int{3}, []float32{1, 2}); err == nil {
		t.Fatal("shape mismatch accepted")
	}
	if _, err := SummarizeFixtureFloat32Tensor("bad", []int{}, nil); err == nil {
		t.Fatal("empty shape accepted")
	}
	if _, err := SummarizeFixtureFloat32Tensor("bad", []int{0}, nil); err == nil {
		t.Fatal("zero dimension accepted")
	}
}

func TestFixtureShapeNumel(t *testing.T) {
	got, err := FixtureShapeNumel([]int{1, 3, 4, 5})
	if err != nil {
		t.Fatal(err)
	}
	if got != 60 {
		t.Fatalf("numel=%d", got)
	}
}

func TestFixtureSummaryRejectsNonfiniteAndStreamsHash(t *testing.T) {
	for _, v := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if _, err := SummarizeFixtureFloat32Tensor("bad", []int{1}, []float32{v}); err == nil {
			t.Fatal("nonfinite fixture accepted")
		}
	}
	vals := make([]float32, 1025)
	raw := make([]byte, len(vals)*4)
	for i := range vals {
		vals[i] = float32(i) * .125
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(vals[i]))
	}
	got, err := SummarizeFixtureFloat32Tensor("long", []int{len(vals)}, vals)
	if err != nil {
		t.Fatal(err)
	}
	want := fmt.Sprintf("%x", sha256.Sum256(raw))
	if got.SHA256LEF32 != want {
		t.Fatal(got.SHA256LEF32, want)
	}
}
func BenchmarkFixtureSummary(b *testing.B) {
	values := make([]float32, 1<<20)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := SummarizeFixtureFloat32Tensor("large", []int{len(values)}, values); err != nil {
			b.Fatal(err)
		}
	}
}
