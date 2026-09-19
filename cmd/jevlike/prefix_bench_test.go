package main

import (
	"bytes"
	"math"
	"testing"
)

func TestPrefixConditionalNearTiesAndShift(t *testing.T) {
	for _, row := range [][]float32{{1, 1}, {1, math.Nextafter32(1, 2)}, {1000, 999, -1000}} {
		p := prefixConditional(row)
		var sum float64
		for _, v := range p {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatal("nonfinite probability")
			}
			sum += v
		}
		if math.Abs(sum-1) > 1e-12 {
			t.Fatal("normalisation")
		}
	}
	p := prefixConditional([]float32{1, 1})
	if p[0] != .5 || p[1] != .5 {
		t.Fatal("strict tie")
	}
	p = prefixConditional([]float32{1, math.Nextafter32(1, 2)})
	if p[1] <= p[0] {
		t.Fatal("one-ULP near tie lost")
	}
	// These are arithmetic fixtures, not observed near-tie model behaviour.
	var out, stderr bytes.Buffer
	if e := run([]string{"prefix-bench", "-h"}, &out, &stderr); e != nil {
		t.Fatal(e)
	}
	if e := run([]string{"prefix-bench"}, &out, &stderr); e == nil {
		t.Fatal("paths required")
	}
}
