package main

import (
	"math"
	"reflect"
	"testing"
)

func TestJoinChunkWaves(t *testing.T) {
	a := []float32{1, 1, 1, 1}
	b := []float32{.5, .5, .5, .5}
	out, err := joinChunkWaves([][]float32{a, b}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0, 1, 1, 0, 0, 0, 0, .5, .5, 0}
	if !reflect.DeepEqual(out, want) {
		t.Fatal(out)
	}
	if a[0] != 1 || b[0] != .5 {
		t.Fatal("mutated source")
	}
	for _, waves := range [][][]float32{nil, {nil}, {{float32(math.NaN())}}} {
		if _, err := joinChunkWaves(waves, 2, 2); err == nil {
			t.Fatal("accepted invalid chunks")
		}
	}
}
