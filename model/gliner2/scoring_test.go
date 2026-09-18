package gliner2

import (
	"math"
	"testing"
)

func TestIntervalPrefix(t *testing.T) {
	mean := float32(2)
	for _, tc := range []struct {
		s, e int
		want float32
	}{{1, 3, 12}, {-4, 20, 15}, {2, 2, 0}, {3, 1, -12}} {
		got, err := IntervalPrefixScore([]float32{0, 1, 4, 9}, tc.s, tc.e, &mean)
		if err != nil || got != tc.want {
			t.Fatalf("%+v got %g err %v", tc, got, err)
		}
	}
	if _, err := IntervalPrefixScore(nil, 0, 1, nil); err == nil {
		t.Fatal("empty prefix accepted")
	}
}
func TestLengthFeatures(t *testing.T) {
	got, err := ContinuousLengthFeatures(2, 6, 8)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(got[0])-math.Log(5)) > 1e-6 || got[1] != .5 || got[2] != .5 {
		t.Fatal(got)
	}
	got, _ = ContinuousLengthFeatures(2, 2, 0)
	if got[1] != 1 || got[2] != 1 {
		t.Fatal(got)
	}
}
func TestEndpointSIMD(t *testing.T) {
	for n := 1; n < 137; n++ {
		a, b := make([]float32, n), make([]float32, n)
		var want float64
		for i := range a {
			a[i] = float32(i%7) - 3
			b[i] = float32(i%9) * .25
			want += float64(a[i]) * float64(b[i])
		}
		got, err := EndpointCompatibility(a, b)
		if err != nil || math.Abs(float64(got)-want) > 1e-5 {
			t.Fatalf("n=%d got %g want %g", n, got, want)
		}
	}
	if _, err := EndpointCompatibility([]float32{1}, nil); err == nil {
		t.Fatal("bad shape")
	}
	if Sigmoid(1000) != 1 || Sigmoid(-1000) != 0 || Sigmoid(0) != .5 {
		t.Fatal("sigmoid extremes")
	}
}
