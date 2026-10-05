package whisper

import (
	"context"
	"math"
	"reflect"
	"testing"
)

// Adopted cross K/V must give the same head-major state as CPU-computed K/V.
func TestDecoderStateFromCrossMatchesCPU(t *testing.T) {
	cfg, dec, enc, frames := syntheticDecoderStateFixture()
	fillSyntheticDecoderWeights(cfg, dec)
	ctx := context.Background()
	want, err := NewDecoderStateContext(ctx, cfg, enc, frames, dec)
	if err != nil {
		t.Fatal(err)
	}
	k := make([][]float32, cfg.DecoderLayers)
	v := make([][]float32, cfg.DecoderLayers)
	for l := range k {
		k[l] = append([]float32(nil), want.CrossK[l]...)
		v[l] = append([]float32(nil), want.CrossV[l]...)
	}
	got, err := newDecoderStateFromCrossContext(ctx, cfg, k, v, frames)
	if err != nil {
		t.Fatal(err)
	}
	for l := range k {
		for _, pair := range [][2][]float32{{got.CrossKHead[l], want.CrossKHead[l]}, {got.CrossVHead[l], want.CrossVHead[l]}} {
			if len(pair[0]) != len(pair[1]) {
				t.Fatal("head length", l)
			}
			for i := range pair[0] {
				if math.Float32bits(pair[0][i]) != math.Float32bits(pair[1][i]) {
					t.Fatal("head bits", l, i)
				}
			}
		}
	}
	tok := 1
	a := dec.ForwardToken(tok, want)
	b := dec.ForwardToken(tok, got)
	if len(a) == 0 || len(a) != len(b) {
		t.Fatal("logit length")
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			t.Fatal("logit bits", i)
		}
	}
	if _, err := newDecoderStateFromCrossContext(ctx, cfg, k[:len(k)-1], v, frames); err == nil {
		t.Fatal("accepted missing layer")
	}
	short := append([][]float32(nil), k...)
	short[0] = short[0][1:]
	if _, err := newDecoderStateFromCrossContext(ctx, cfg, short, v, frames); err == nil {
		t.Fatal("accepted short layer")
	}
}

func TestVulkanEncoderCrossKVLayout(t *testing.T) {
	cfg := Config{EncoderDModel: 4, DecoderDModel: 4, DecoderLayers: 2}
	base := func() *vkEncoderLayout {
		return &vkEncoderLayout{cfg: cfg, rows: 3, weights: [][]vkEncoderTensor{{{name: "final.w", shape: []int{4}}}},
			scratch: []vkEncoderTensor{{name: "h", shape: []int{3, 4}}},
			plans:   [][]vkEncoderStep{{{op: "add", out: "x", in: []string{"a", "b"}}}, {{op: "norm", out: "h", in: []string{"h", "final.w", "final.b"}}}}}
	}
	dec := &Decoder{Layers: make([]DecoderLayer, 2)}
	for l := range dec.Layers {
		dec.Layers[l].CrossVBias = []float32{1, 2, 3, float32(l)}
	}
	dec.Layers[1].CrossKBias = []float32{4, 5, 6, 7}
	layout := base()
	original := layout.plans
	if err := addVulkanEncoderCrossKV(layout, dec, vulkanLinearOriginalQ5PaddedIntegerDotMMQTanh, true); err != nil {
		t.Fatal(err)
	}
	if len(original[1]) != 1 || !reflect.DeepEqual(layout.plans[0], original[0]) {
		t.Fatal("source plans mutated")
	}
	final := layout.plans[1]
	var outs []string
	for _, s := range final[1:] {
		if s.op != "linear" || s.in[0] != "h" {
			t.Fatalf("step %+v", s)
		}
		outs = append(outs, s.out+":"+s.in[1]+":"+s.in[2])
	}
	if want := []string{"xk0:dec0.xk.w:dec0.xk.b", "xv0:dec0.xv.w:dec0.xv.b", "xk1:dec1.xk.w:dec1.xk.b", "xv1:dec1.xv.w:dec1.xv.b"}; !reflect.DeepEqual(outs, want) {
		t.Fatal(outs)
	}
	group := layout.weights[len(layout.weights)-1]
	byName := map[string]vkEncoderTensor{}
	for _, w := range group {
		byName[w.name] = w
	}
	if w := byName["dec0.xk.b"]; !w.zero || w.data != nil {
		t.Fatal("absent key bias must be zero")
	}
	if w := byName["dec1.xk.b"]; w.zero || !reflect.DeepEqual(w.data, dec.Layers[1].CrossKBias) {
		t.Fatal("key bias data")
	}
	if w := byName["dec1.xv.b"]; !reflect.DeepEqual(w.data, dec.Layers[1].CrossVBias) {
		t.Fatal("value bias data")
	}
	if w := byName["dec0.xk.w"]; !reflect.DeepEqual(w.shape, []int{4, 4}) || w.data != nil {
		t.Fatal("packed weight spec")
	}
	if len(layout.scratch) != 5 || layout.scratch[1].name != "xk0" || !reflect.DeepEqual(layout.scratch[4].shape, []int{3, 4}) {
		t.Fatal("scratch", layout.scratch)
	}
	for name, mutate := range map[string]func(*vkEncoderLayout, *Decoder) (vulkanLinearMode, bool){
		"mode": func(*vkEncoderLayout, *Decoder) (vulkanLinearMode, bool) {
			return vulkanLinearOriginalQ5PaddedIntegerDotMMQ, true
		},
		"unpacked": func(*vkEncoderLayout, *Decoder) (vulkanLinearMode, bool) {
			return vulkanLinearOriginalQ5PaddedIntegerDotMMQTanh, false
		},
		"layers": func(_ *vkEncoderLayout, d *Decoder) (vulkanLinearMode, bool) {
			d.Layers = d.Layers[:1]
			return vulkanLinearOriginalQ5PaddedIntegerDotMMQTanh, true
		},
		"value bias": func(_ *vkEncoderLayout, d *Decoder) (vulkanLinearMode, bool) {
			d.Layers[0].CrossVBias = nil
			return vulkanLinearOriginalQ5PaddedIntegerDotMMQTanh, true
		},
		"key bias": func(_ *vkEncoderLayout, d *Decoder) (vulkanLinearMode, bool) {
			d.Layers[0].CrossKBias = []float32{1}
			return vulkanLinearOriginalQ5PaddedIntegerDotMMQTanh, true
		},
		"final plan": func(l *vkEncoderLayout, _ *Decoder) (vulkanLinearMode, bool) {
			l.plans[1][0].op = "add"
			return vulkanLinearOriginalQ5PaddedIntegerDotMMQTanh, true
		},
	} {
		l := base()
		d := &Decoder{Layers: append([]DecoderLayer(nil), dec.Layers...)}
		mode, packed := mutate(l, d)
		if err := addVulkanEncoderCrossKV(l, d, mode, packed); err == nil {
			t.Fatal("accepted", name)
		}
	}
}
