package training

import (
	"math"
	"testing"
)

func TestTrainingDualSiLUFiniteDifference(t *testing.T) {
	max, square := float64(0), float64(0)
	count := 0
	for _, x := range []float64{-10, -2, -.1, 0, .1, 2, 10} {
		for _, xt := range []float64{-.3, .4} {
			gv, gt := .7, -.2
			objective := func(v, tangent float64) float64 {
				s := 1 / (1 + math.Exp(-v))
				return gv*v*s + gt*(s*(1+v*(1-s)))*tangent
			}
			epsilon := 1e-5
			wantV := (objective(x+epsilon, xt) - objective(x-epsilon, xt)) / (2 * epsilon)
			wantT := (objective(x, xt+epsilon) - objective(x, xt-epsilon)) / (2 * epsilon)
			xf, tf, gf, gtf := float32(x), float32(xt), float32(gv), float32(gt)
			e := float32(math.Exp(-math.Abs(x)))
			s := float32(1) / (1 + e)
			if xf < 0 {
				s = e / (1 + e)
			}
			first := s * (1 + xf*(1-s))
			second := s * (1 - s) * (2 + xf*(1-2*s))
			gotV, gotT := gf*first+(gtf*second)*tf, gtf*first
			for _, pair := range [][2]float64{{float64(gotV), wantV}, {float64(gotT), wantT}} {
				err := math.Abs(pair[0] - pair[1])
				max = math.Max(max, err)
				square += err * err
				count++
				if err > 2e-6 {
					t.Fatal(x, xt, pair)
				}
			}
		}
	}
	t.Logf("dual SiLU host schedule: values=%d max_abs=%g RMS=%g", count, max, math.Sqrt(square/float64(count)))
}
