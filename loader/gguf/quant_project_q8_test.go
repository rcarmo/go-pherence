package gguf

import (
	"math/rand"
	"runtime"
	"testing"
)

func TestQuantizeQ8_0BatchParallelExact(t *testing.T) {
	const batch, width = 124, 256
	rng := rand.New(rand.NewSource(0x8040124))
	x := make([]float32, batch*width)
	for i := range x {
		x[i] = (rng.Float32()*2 - 1) * 8
	}
	want := make([]q8_0Block, batch*width/qk8_0)
	if err := quantizeQ8_0To(want, x); err != nil {
		t.Fatal(err)
	}
	got := make([]q8_0Block, len(want))
	previous := runtime.GOMAXPROCS(6)
	defer runtime.GOMAXPROCS(previous)
	if err := quantizeQ8_0BatchTo(got, x, batch, width); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("block %d differs: got=%+v want=%+v", i, got[i], want[i])
		}
	}
}

func TestQuantizeQ8_0BatchRejectsMalformedShape(t *testing.T) {
	if err := quantizeQ8_0BatchTo(make([]q8_0Block, 1), make([]float32, 32), 64, 32); err == nil {
		t.Fatal("expected malformed batch error")
	}
}
