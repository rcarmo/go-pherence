package nvfp4

import (
	"math"
	"testing"
)

func TestDecodeFP4E2M1Codebook(t *testing.T) {
	want := []float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6}
	for code, w := range want {
		got := DecodeFP4E2M1(byte(code))
		if got != w {
			t.Fatalf("DecodeFP4E2M1(%#x)=%v want %v", code, got, w)
		}
	}
}

func TestDecodeF8E4M3(t *testing.T) {
	cases := []struct {
		name string
		code byte
		want float32
	}{
		{name: "positive zero", code: 0x00, want: 0},
		{name: "negative zero", code: 0x80, want: float32(math.Copysign(0, -1))},
		{name: "min positive subnormal", code: 0x01, want: 1.0 / 512},
		{name: "negative subnormal", code: 0x81, want: -1.0 / 512},
		{name: "largest subnormal", code: 0x07, want: 7.0 / 512},
		{name: "min normal", code: 0x08, want: 1.0 / 64},
		{name: "one", code: 0x38, want: 1},
		{name: "one point five", code: 0x3c, want: 1.5},
		{name: "two", code: 0x40, want: 2},
		{name: "large normal", code: 0x78, want: 256},
		{name: "largest finite", code: 0x7e, want: 448},
		{name: "negative normal", code: 0xb8, want: -1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := DecodeF8E4M3(tc.code)
			if math.Abs(float64(got-tc.want)) > 1e-7 || math.Signbit(float64(got)) != math.Signbit(float64(tc.want)) {
				t.Fatalf("DecodeF8E4M3(%#x)=%v want %v", tc.code, got, tc.want)
			}
		})
	}
	for _, code := range []byte{0x7f, 0xff} {
		if got := DecodeF8E4M3(code); !math.IsNaN(float64(got)) {
			t.Fatalf("DecodeF8E4M3(%#x)=%v, want NaN", code, got)
		}
	}
}

func TestUnpackNVFP4RejectsOutOfRangeCountWithoutOverflow(t *testing.T) {
	if got := UnpackNVFP4([]byte{0x12}, 3); got != nil {
		t.Fatalf("UnpackNVFP4 count beyond packed bytes=%v want nil", got)
	}
	if got := UnpackNVFP4(nil, int(^uint(0)>>1)); got != nil {
		t.Fatalf("UnpackNVFP4 huge count len=%d want nil", len(got))
	}
}

func TestUnpackNVFP4LowNibbleFirst(t *testing.T) {
	got := UnpackNVFP4([]byte{0x10, 0x32, 0xba}, 6)
	want := []float32{0, 0.5, 1, 1.5, -1, -1.5}
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestDequantNVFP4Synthetic(t *testing.T) {
	qw := syntheticNVFP4Weight()
	got := DequantNVFP4(qw)
	want := []float32{0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 0, -0.25, -0.5, -0.75, -1, -1.5, -2, -3}
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	gotTo := make([]float32, len(got)+1)
	gotTo[len(got)] = 123
	if !DequantNVFP4To(gotTo, qw) {
		t.Fatal("DequantNVFP4To returned false for valid weight")
	}
	if gotTo[len(got)] != 123 {
		t.Fatalf("DequantNVFP4To mutated tail: %v", gotTo)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%v want %v", i, got[i], want[i])
		}
		if gotTo[i] != want[i] {
			t.Fatalf("DequantNVFP4To got[%d]=%v want %v", i, gotTo[i], want[i])
		}
	}
}

func TestGemvNVFP4MatchesDequantizedReference(t *testing.T) {
	for _, qw := range []*NVFP4Weight{syntheticNVFP4Weight(), syntheticOddGroupNVFP4Weight()} {
		t.Run("groupSize", func(t *testing.T) {
			x := []float32{1, -1, 2, -2, 0.5, -0.5, 3, -3, 1, 1, 1, 1, -1, -1, -1, -1}
			x = x[:qw.InDim]
			wantWeights := DequantNVFP4(qw)
			want := make([]float32, qw.OutDim)
			for row := 0; row < qw.OutDim; row++ {
				for col := 0; col < qw.InDim; col++ {
					want[row] += wantWeights[row*qw.InDim+col] * x[col]
				}
			}
			got := make([]float32, qw.OutDim)
			if !GemvNVFP4To(got, x, qw) {
				t.Fatal("GemvNVFP4To returned false for valid weight")
			}
			ref := make([]float32, qw.OutDim)
			if !GemvNVFP4Reference(ref, x, qw) {
				t.Fatal("GemvNVFP4Reference returned false for valid weight")
			}
			for i := range want {
				if math.Abs(float64(got[i]-want[i])) > 1e-6 {
					t.Fatalf("GemvNVFP4[%d]=%v want %v", i, got[i], want[i])
				}
				if math.Abs(float64(ref[i]-want[i])) > 1e-6 {
					t.Fatalf("GemvNVFP4Reference[%d]=%v want %v", i, ref[i], want[i])
				}
			}
		})
	}
}

func TestDequantNVFP4ParallelMatchesReference(t *testing.T) {
	qw := &NVFP4Weight{
		OutDim:       1024,
		InDim:        16,
		Groups:       1,
		GroupSize:    16,
		Weight:       make([]byte, 1024*8),
		WeightScale:  make([]byte, 1024),
		WeightScale2: 0.5,
	}
	pattern := []byte{0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe}
	for row := 0; row < qw.OutDim; row++ {
		copy(qw.Weight[row*8:(row+1)*8], pattern)
		qw.WeightScale[row] = 0x38 // 1.0
	}
	got := make([]float32, qw.OutDim*qw.InDim+1)
	got[len(got)-1] = 123
	if !DequantNVFP4To(got, qw) {
		t.Fatal("DequantNVFP4To returned false")
	}
	for row := 0; row < qw.OutDim; row++ {
		for col := 0; col < qw.InDim; col++ {
			want := nvfp4At(qw, row, col)
			if got[row*qw.InDim+col] != want {
				t.Fatalf("got[%d,%d]=%v want %v", row, col, got[row*qw.InDim+col], want)
			}
		}
	}
	if got[len(got)-1] != 123 {
		t.Fatalf("DequantNVFP4To mutated tail")
	}
}

func TestGemmNVFP4MatchesRepeatedGemv(t *testing.T) {
	qw := syntheticNVFP4Weight()
	batch := 3
	x := []float32{
		1, -1, 2, -2, 0.5, -0.5, 3, -3, 4, -4, 1.5, -1.5, 2.5, -2.5, 0.25, -0.25,
		0.25, -0.5, 1.5, -2, 0.75, -1.25, 2.5, -3, 1, 0.5, -0.75, 1.25, -1.5, 2, -2.5, 3,
		-1, 2, -3, 4, -0.5, 0.5, -3, 3, -4, 4, -1.5, 1.5, -2.5, 2.5, -0.25, 0.25,
	}
	got := make([]float32, batch*qw.OutDim+1)
	got[len(got)-1] = 123
	if !GemmNVFP4(got, x, batch, qw) {
		t.Fatal("GemmNVFP4 returned false for valid input")
	}
	want := make([]float32, batch*qw.OutDim)
	for b := 0; b < batch; b++ {
		if !GemvNVFP4Reference(want[b*qw.OutDim:(b+1)*qw.OutDim], x[b*qw.InDim:(b+1)*qw.InDim], qw) {
			t.Fatal("GemvNVFP4Reference returned false")
		}
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("got[%d]=%v want %v", i, got[i], want[i])
		}
	}
	if got[len(got)-1] != 123 {
		t.Fatal("GemmNVFP4 mutated tail")
	}
}

func TestGemmNVFP4RejectsMalformedInputs(t *testing.T) {
	qw := syntheticNVFP4Weight()
	if GemmNVFP4(make([]float32, qw.OutDim), make([]float32, qw.InDim), 0, qw) {
		t.Fatal("GemmNVFP4 accepted zero batch")
	}
	if GemmNVFP4(make([]float32, qw.OutDim), make([]float32, qw.InDim-1), 1, qw) {
		t.Fatal("GemmNVFP4 accepted short x")
	}
	if GemmNVFP4(make([]float32, qw.OutDim-1), make([]float32, qw.InDim), 1, qw) {
		t.Fatal("GemmNVFP4 accepted short out")
	}
	if GemmNVFP4(make([]float32, qw.OutDim), make([]float32, qw.InDim), 1, nil) {
		t.Fatal("GemmNVFP4 accepted nil weight")
	}
}

func TestGemvNVFP4ToRejectsMalformedInputs(t *testing.T) {
	qw := syntheticNVFP4Weight()
	if !GemvNVFP4To(make([]float32, qw.OutDim), make([]float32, qw.InDim), qw) {
		t.Fatal("GemvNVFP4To returned false for valid input")
	}
	if GemvNVFP4To(make([]float32, qw.OutDim-1), make([]float32, qw.InDim), qw) {
		t.Fatal("GemvNVFP4To accepted short output")
	}
	if GemvNVFP4To(make([]float32, qw.OutDim), make([]float32, qw.InDim-1), qw) {
		t.Fatal("GemvNVFP4To accepted short x")
	}
	if GemvNVFP4To(make([]float32, qw.OutDim), make([]float32, qw.InDim), nil) {
		t.Fatal("GemvNVFP4To accepted nil weight")
	}
}

func TestNVFP4TinySyntheticLogitsMatchF32Reference(t *testing.T) {
	qw := syntheticNVFP4LogitWeight()
	hidden := []float32{0.25, -0.5, 1.5, -2, 0.75, -1.25, 2.5, -3, 1, 0.5, -0.75, 1.25, -1.5, 2, -2.5, 3}

	got := make([]float32, qw.OutDim)
	GemvNVFP4(got, hidden, qw)

	weights := DequantNVFP4(qw)
	want := make([]float32, qw.OutDim)
	for row := 0; row < qw.OutDim; row++ {
		for col := 0; col < qw.InDim; col++ {
			want[row] += weights[row*qw.InDim+col] * hidden[col]
		}
	}

	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("logit[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func syntheticNVFP4Weight() *NVFP4Weight {
	return &NVFP4Weight{
		// Low nibble first: codes 0..15 across one 16-value group.
		Weight:       []byte{0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe},
		WeightScale:  []byte{0x40}, // E4M3 2.0
		WeightScale2: 0.25,
		OutDim:       1,
		InDim:        16,
		Groups:       1,
		GroupSize:    16,
	}
}

func syntheticOddGroupNVFP4Weight() *NVFP4Weight {
	return &NVFP4Weight{
		// Six values split into two odd-sized groups. This catches nibble selection
		// when a group starts on a high nibble rather than a byte boundary.
		Weight:       []byte{0x10, 0x32, 0x54},
		WeightScale:  []byte{0x38, 0x40}, // 1.0, 2.0
		WeightScale2: 0.5,
		OutDim:       1,
		InDim:        6,
		Groups:       2,
		GroupSize:    3,
	}
}

func syntheticNVFP4LogitWeight() *NVFP4Weight {
	return &NVFP4Weight{
		// Three vocab rows, each one 16-value group. The bytes deliberately mix
		// positive/negative FP4 codes so logits exercise signs and scale handling.
		Weight: []byte{
			0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
			0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67,
			0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x0f,
		},
		WeightScale: []byte{
			0x38, // 1.0
			0x40, // 2.0
			0x34, // 0.75
		},
		WeightScale2: 0.5,
		OutDim:       3,
		InDim:        16,
		Groups:       1,
		GroupSize:    16,
	}
}

func TestValidateNVFP4WeightObservedLayouts(t *testing.T) {
	cases := []struct {
		name      string
		outDim    int
		packedIn  int
		scaleCols int
	}{
		{"qwen3 dense q_proj", 4096, 2048, 256},
		{"qwen3 dense down_proj", 4096, 6144, 768},
		{"qwen3 moe expert down_proj", 2048, 384, 48},
		{"gemma4 dense down_proj", 5376, 10752, 1344},
		{"gemma4 moe expert scale", 2816, 352, 44},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			qw := &NVFP4Weight{
				Weight:      make([]byte, tc.outDim*tc.packedIn),
				WeightScale: make([]byte, tc.outDim*tc.scaleCols),
				OutDim:      tc.outDim,
				InDim:       tc.packedIn * 2,
				Groups:      tc.scaleCols,
				GroupSize:   16,
			}
			if err := ValidateNVFP4Weight(qw); err != nil {
				t.Fatalf("ValidateNVFP4Weight: %v", err)
			}
		})
	}
}

func TestValidateNVFP4WeightRejectsBadLayout(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	cases := []struct {
		name string
		qw   *NVFP4Weight
	}{
		{"nil", nil},
		{"odd input", &NVFP4Weight{Weight: make([]byte, 1), WeightScale: make([]byte, 1), OutDim: 1, InDim: 3, Groups: 1, GroupSize: 3}},
		{"bad groups", &NVFP4Weight{Weight: make([]byte, 8), WeightScale: make([]byte, 2), OutDim: 2, InDim: 8, Groups: 2, GroupSize: 8}},
		{"short weight", &NVFP4Weight{Weight: make([]byte, 7), WeightScale: make([]byte, 2), OutDim: 2, InDim: 8, Groups: 1, GroupSize: 8}},
		{"short scale", &NVFP4Weight{Weight: make([]byte, 8), WeightScale: make([]byte, 1), OutDim: 2, InDim: 8, Groups: 1, GroupSize: 8}},
		{"weight overflow", &NVFP4Weight{OutDim: maxInt/2 + 1, InDim: 4, Groups: 1, GroupSize: 4}},
		{"scale overflow", &NVFP4Weight{OutDim: maxInt/2 + 1, InDim: 2, Groups: 2, GroupSize: 1}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := ValidateNVFP4Weight(tc.qw); err == nil {
				t.Fatal("ValidateNVFP4Weight succeeded, want error")
			}
			if got := DequantNVFP4(tc.qw); got != nil {
				t.Fatalf("DequantNVFP4 malformed len=%d, want nil", len(got))
			}
		})
	}
}
