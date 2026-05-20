package format

import (
	"strings"
	"testing"
)

func validGPTQInputs() ([]int32, []int32, []int32, []float32, int, int) {
	inFeatures, outFeatures := 8, 8
	qweight := make([]int32, (inFeatures/8)*outFeatures)
	qzeros := make([]int32, 1*(outFeatures/8))
	gIdx := make([]int32, inFeatures)
	scales := make([]float32, 1*outFeatures)
	return qweight, qzeros, gIdx, scales, inFeatures, outFeatures
}

func TestValidateAcceptsValidInputs(t *testing.T) {
	qw, qz, g, s, in, out := validGPTQInputs()
	if err := Validate(qw, qz, g, s, in, out, false); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if err := ValidateSym(qw, g, s, in, out); err != nil {
		t.Fatalf("ValidateSym: %v", err)
	}
	if got := Dequant(qw, qz, g, s, in, out, false); len(got) != in*out {
		t.Fatalf("Dequant len=%d, want %d", len(got), in*out)
	}
	if got := DequantSym(qw, g, s, in, out); len(got) != in*out {
		t.Fatalf("DequantSym len=%d, want %d", len(got), in*out)
	}
}

func TestValidateRejectsMalformedInputs(t *testing.T) {
	qw, qz, g, s, in, out := validGPTQInputs()
	cases := []struct {
		name string
		fn   func() error
		want string
	}{
		{name: "bad in", fn: func() error { return Validate(qw, qz, g, s, 7, out, false) }, want: "divisible"},
		{name: "bad out", fn: func() error { return Validate(qw, qz, g, s, in, 7, false) }, want: "outFeatures"},
		{name: "short qweight", fn: func() error { return Validate(qw[:1], qz, g, s, 16, out, false) }, want: "qweight"},
		{name: "short gidx", fn: func() error { return Validate(qw, qz, g[:1], s, in, out, false) }, want: "g_idx"},
		{name: "negative group", fn: func() error {
			bad := append([]int32(nil), g...)
			bad[0] = -1
			return Validate(qw, qz, bad, s, in, out, false)
		}, want: "negative"},
		{name: "short scales", fn: func() error {
			bad := append([]int32(nil), g...)
			bad[0] = 1
			return Validate(qw, qz, bad, s, in, out, false)
		}, want: "scales"},
		{name: "short qzeros", fn: func() error {
			bad := append([]int32(nil), g...)
			bad[0] = 1
			ss := make([]float32, 16)
			return Validate(qw, qz, bad, ss, in, out, false)
		}, want: "qzeros"},
		{name: "output overflow", fn: func() error {
			maxInt := int(^uint(0) >> 1)
			return Validate(nil, nil, nil, nil, maxInt/8*8, 16, true)
		}, want: "output size overflows"},
		{name: "gemv q4 negative dims", fn: func() error {
			return ValidateGemvSym(nil, nil, nil, nil, nil, -8, out)
		}, want: "invalid GPTQ dims"},
		{name: "gemv q4 output overflow", fn: func() error {
			maxInt := int(^uint(0) >> 1)
			return ValidateGemvSym(nil, nil, nil, nil, nil, maxInt/8*8, 16)
		}, want: "output size overflows"},
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

func TestFloat16ToFloat32HandlesSubnormalAndSpecials(t *testing.T) {
	if got := Float16ToFloat32(0x0001); got != float32(1.0/(1<<24)) {
		t.Fatalf("min subnormal=%g", got)
	}
	if got := Float16ToFloat32(0x8001); got != -float32(1.0/(1<<24)) {
		t.Fatalf("negative min subnormal=%g", got)
	}
	if got := Float16ToFloat32(0x3c00); got != 1 {
		t.Fatalf("one=%g", got)
	}
}

func TestDequantKnownValues(t *testing.T) {
	inFeatures, outFeatures := 8, 2
	packed0 := uint32(0xFEDCBA98)
	packed1 := uint32(0x76543210)
	qweight := []int32{int32(packed0), int32(packed1)}
	gIdx := make([]int32, inFeatures)
	scales := []float32{0.5, 2.0}
	got := DequantSym(qweight, gIdx, scales, inFeatures, outFeatures)
	if len(got) != inFeatures*outFeatures {
		t.Fatalf("len=%d", len(got))
	}
	// output row 0 uses packed nibbles 8..15 with scale 0.5.
	for i := 0; i < inFeatures; i++ {
		want := float32(i) * 0.5
		if got[i] != want {
			t.Fatalf("row0[%d]=%g want %g", i, got[i], want)
		}
	}
	// output row 1 uses packed nibbles 0..7 with scale 2.
	row1 := got[inFeatures:]
	for i := 0; i < inFeatures; i++ {
		want := float32(i-8) * 2
		if row1[i] != want {
			t.Fatalf("row1[%d]=%g want %g", i, row1[i], want)
		}
	}
}

func TestDequantRejectsMalformedInputsWithoutPanic(t *testing.T) {
	qw, qz, g, s, _, out := validGPTQInputs()
	if got := Dequant(qw[:1], qz, g, s, 16, out, false); got != nil {
		t.Fatalf("Dequant malformed len=%d, want nil", len(got))
	}
	if got := DequantSym(qw[:1], g, s, 16, out); got != nil {
		t.Fatalf("DequantSym malformed len=%d, want nil", len(got))
	}
}

func TestDequantRejectsOutputOverflow(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	in, out := maxInt/8*8, 16
	if got := Dequant(nil, nil, nil, nil, in, out, true); got != nil {
		t.Fatalf("Dequant overflow len=%d, want nil", len(got))
	}
	if got := DequantSym(nil, nil, nil, in, out); got != nil {
		t.Fatalf("DequantSym overflow len=%d, want nil", len(got))
	}
}
