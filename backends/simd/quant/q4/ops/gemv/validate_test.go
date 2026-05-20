package gemv

import (
	"strings"
	"testing"
)

func validGemvQ4Inputs() ([]float32, []float32, []int32, []int32, []float32, int, int) {
	inDim, outDim := 8, 2
	out := make([]float32, outDim)
	x := make([]float32, inDim)
	qweight := make([]int32, (inDim/8)*outDim)
	gIdx := make([]int32, inDim)
	scales := make([]float32, outDim)
	return out, x, qweight, gIdx, scales, inDim, outDim
}

func TestValidateGemvSym(t *testing.T) {
	out, x, qw, g, s, inDim, outDim := validGemvQ4Inputs()
	if err := ValidateGemvSym(out, x, qw, g, s, inDim, outDim); err != nil {
		t.Fatalf("ValidateGemvSym: %v", err)
	}
	cases := []struct {
		name string
		fn   func() error
		want string
	}{
		{name: "short out", fn: func() error { return ValidateGemvSym(out[:1], x, qw, g, s, inDim, outDim) }, want: "out length"},
		{name: "short x", fn: func() error { return ValidateGemvSym(out, x[:1], qw, g, s, inDim, outDim) }, want: "x length"},
		{name: "short qweight", fn: func() error { return ValidateGemvSym(out, x, nil, g, s, inDim, outDim) }, want: "qweight"},
		{name: "bad group", fn: func() error {
			bad := append([]int32(nil), g...)
			bad[0] = 1
			return ValidateGemvSym(out, x, qw, bad, s, inDim, outDim)
		}, want: "scales"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.fn()
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestGemvSymMalformedDoesNotPanic(t *testing.T) {
	out, x, qw, g, s, inDim, outDim := validGemvQ4Inputs()
	out[0] = 123
	GemvSym(out, x, qw[:1], g, s, 16, outDim)
	if out[0] != 123 {
		t.Fatalf("malformed GemvSym should leave output unchanged, got %f", out[0])
	}
	GemvSym(out[:1], x, qw, g, s, inDim, outDim)
}
