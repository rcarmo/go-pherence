package training

import (
	"math"
	"testing"
)

func TestAdamWStepAdmission(t *testing.T) {
	p, err := NewAdamWStep(.01, .9, .999, 1e-8, .01, .9, 1, 1)
	if err != nil || p.Bias1 != float32(1-float64(float32(.9))) || p.Bias2 != float32(1-float64(float32(.999))) {
		t.Fatal(p, err)
	}
	for _, step := range []uint64{0, math.MaxUint64} {
		if _, err := NewAdamWStep(.01, .9, .999, 1e-8, 0, 0, 1, step); err == nil {
			t.Fatal("step admitted")
		}
	}
	for i := 0; i < 9; i++ {
		q := p
		v := []*float32{&q.LearningRate, &q.Beta1, &q.Beta2, &q.Epsilon, &q.WeightDecay, &q.Bias1, &q.Bias2, &q.EMADecay, &q.GradientScale}
		*v[i] = float32(math.NaN())
		if q.Validate() == nil {
			t.Fatal("NaN admitted", i)
		}
	}
	for _, mutate := range []func(*AdamWStep){func(q *AdamWStep) { q.LearningRate = 0 }, func(q *AdamWStep) { q.Beta1 = 1 }, func(q *AdamWStep) { q.Beta2 = -1 }, func(q *AdamWStep) { q.Epsilon = 0 }, func(q *AdamWStep) { q.WeightDecay = -1 }, func(q *AdamWStep) { q.Bias1 = 0 }, func(q *AdamWStep) { q.Bias2 = 2 }, func(q *AdamWStep) { q.EMADecay = 1 }, func(q *AdamWStep) { q.GradientScale = 2 }, func(q *AdamWStep) { q.WeightDecay = math.MaxFloat32; q.LearningRate = math.MaxFloat32 }} {
		q := p
		mutate(&q)
		if q.Validate() == nil {
			t.Fatal("invalid constants admitted", q)
		}
	}
}
func TestTrainingLinearShape(t *testing.T) {
	for _, d := range [][3]int{{1, 1, 1}, {1024, 16384, 16384}} {
		if err := LinearShape(d[0], d[1], d[2]); err != nil {
			t.Fatal(err)
		}
	}
	for _, d := range [][3]int{{0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {1025, 1, 1}, {1, 16385, 1}, {1, 1, 16385}} {
		if LinearShape(d[0], d[1], d[2]) == nil {
			t.Fatal("invalid shape")
		}
	}
}
