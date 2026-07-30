package expertstream

import "testing"

func TestQuantSpecLayoutAndSplit(t *testing.T) {
	q := &QuantSpec{OutDim: 4, InDim: 8, GroupSize: 4, Bits: 4}
	layout, err := q.Layout(4*1*4 + 4*2*4 + 4*2*4)
	if err != nil {
		t.Fatal(err)
	}
	if layout.WeightSize != 16 || layout.ScaleSize != 32 || layout.BiasSize != 32 || layout.Groups != 2 {
		t.Fatalf("layout=%+v", layout)
	}
	raw := make([]byte, 80)
	w, s, b, err := q.SplitBytes(raw)
	if err != nil || len(w) != 16 || len(s) != 32 || len(b) != 32 {
		t.Fatalf("split lens=%d/%d/%d err=%v", len(w), len(s), len(b), err)
	}
}

func TestQuantSpecRejectsMalformedLayouts(t *testing.T) {
	cases := []QuantSpec{
		{},
		{OutDim: 4, InDim: 7, GroupSize: 4, Bits: 4},
		{OutDim: 4, InDim: 8, GroupSize: 3, Bits: 4},
		{OutDim: 4, InDim: 8, GroupSize: 4, Bits: 3},
	}
	for _, q := range cases {
		if _, err := q.Layout(80); err == nil {
			t.Fatalf("Layout(%+v) unexpectedly succeeded", q)
		}
	}
}

func TestValidateComponentRequiresQuantDTypePair(t *testing.T) {
	q := &QuantSpec{OutDim: 4, InDim: 8, GroupSize: 4, Bits: 4}
	base := ComponentSpec{Size: 80, Shape: []int64{4, 8}}
	for _, spec := range []ComponentSpec{
		{Offset: 0, Size: base.Size, Shape: base.Shape, DType: DTypeMLXQuant},
		{Offset: 0, Size: base.Size, Shape: base.Shape, DType: "u8", Quant: q},
	} {
		if err := validateComponent(spec, "gate"); err == nil {
			t.Fatalf("validateComponent(%+v) unexpectedly succeeded", spec)
		}
	}
	good := base
	good.DType, good.Quant = DTypeMLXQuant, q
	if err := validateComponent(good, "gate"); err != nil {
		t.Fatal(err)
	}
}
