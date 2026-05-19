package mlx

import (
	"math"
	"testing"
)

func TestFloat16ToFloat32HandlesSubnormalAndSpecials(t *testing.T) {
	if got := float16ToFloat32(0x0001); got != float32(math.Ldexp(1, -24)) {
		t.Fatalf("min subnormal=%g want %g", got, float32(math.Ldexp(1, -24)))
	}
	if got := float16ToFloat32(0x8001); got != -float32(math.Ldexp(1, -24)) {
		t.Fatalf("negative min subnormal=%g", got)
	}
	if got := float16ToFloat32(0x3c00); got != 1 {
		t.Fatalf("one=%g", got)
	}
	if got := float16ToFloat32(0x7c00); !math.IsInf(float64(got), 1) {
		t.Fatalf("+inf=%g", got)
	}
	if got := float16ToFloat32(0xfc00); !math.IsInf(float64(got), -1) {
		t.Fatalf("-inf=%g", got)
	}
	if got := float16ToFloat32(0x7e00); !math.IsNaN(float64(got)) {
		t.Fatalf("nan=%g", got)
	}
}
