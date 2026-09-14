package omnivoice

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
)

func TestDirectQ8BackboneParityReuse(t *testing.T) {
	seed, _ := loadBackboneFixture(t)
	cfg := seed.weights.Config
	cfg.LLMConfig.HiddenSize = 32
	cfg.LLMConfig.HeadDim = 32
	cfg.LLMConfig.NumAttentionHeads = 1
	cfg.LLMConfig.NumKeyValueHeads = 1
	cfg.LLMConfig.IntermediateSize = 64
	shapes := loader.ExpectedShapes(cfg)
	specs := []gguf.TensorSpec{}
	raws := map[string][]byte{}
	for name, s := range shapes {
		if name == "codebook_layer_offsets" {
			continue
		}
		shape := make([]uint64, len(s))
		n := 1
		for i, d := range s {
			shape[len(s)-1-i] = uint64(d)
			n *= int(d)
		}
		qt := gguf.QuantF32
		var raw []byte
		if loader.IsQ8ProjectionName(name) {
			qt = gguf.QuantQ8_0
			raw = make([]byte, n/32*34)
			for block := 0; block < n/32; block++ {
				binary.LittleEndian.PutUint16(raw[block*34:], half.F32ToF16(0.0005))
				for j := 0; j < 32; j++ {
					raw[block*34+2+j] = byte(int8((block+j)%31 - 15))
				}
			}
		} else {
			raw = make([]byte, n*4)
			for i := 0; i < n; i++ {
				v := float32(1)
				if len(s) > 1 {
					v = float32(math.Sin(float64(i))) * 0.02
				}
				binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
			}
		}
		specs = append(specs, gguf.TensorSpec{Name: name, Shape: shape, QType: qt})
		raws[name] = raw
	}
	cfgRaw, _ := json.Marshal(cfg)
	path := filepath.Join(t.TempDir(), "q8.gguf")
	meta := []gguf.MetadataEntry{{Key: "general.architecture", Value: "omnivoice"}, {Key: "omnivoice.schema_version", Value: uint32(1)}, {Key: "omnivoice.config_json", Value: string(cfgRaw)}}
	if err := gguf.WriteV3(context.Background(), path, meta, specs, func(_ context.Context, _ int, s gguf.TensorSpec) (io.Reader, error) {
		return bytes.NewReader(raws[s.Name]), nil
	}); err != nil {
		t.Fatal(err)
	}
	w, err := loader.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	base, err := NewBackbone(w, 7)
	if err != nil {
		t.Fatal(err)
	}
	defer base.Close()
	direct, err := NewBackbone(w, 7)
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	if err := direct.EnableDirectQ8(); err != nil {
		t.Fatal(err)
	}
	if err := direct.EnableResident(context.Background(), 1<<30); err == nil {
		t.Fatal("resident accepted")
	}
	if err := direct.EnableResidentPrepacked(context.Background(), 1<<30); err == nil {
		t.Fatal("prepack accepted")
	}
	sibling, err := NewBackboneSibling(direct, 7)
	if err != nil {
		t.Fatal(err)
	}
	defer sibling.Close()
	if sibling.directQ8 == nil {
		t.Fatal("sibling did not inherit")
	}
	ids := make([]int, 7*cfg.NumAudioCodebook)
	mask := make([]bool, 7)
	for i := range mask {
		mask[i] = true
	}
	a := make([]float32, 7*cfg.NumAudioCodebook*cfg.AudioVocabSize)
	b := make([]float32, len(a))
	ctx := context.Background()
	if err := base.ForwardInto(ctx, a, ids, mask, nil, nil); err != nil {
		t.Fatal(err)
	}
	for _, model := range []*Backbone{direct, sibling, direct} {
		if err := model.ForwardInto(ctx, b, ids, mask, nil, nil); err != nil {
			t.Fatal(err)
		}
		for i := range a {
			if math.Abs(float64(a[i]-b[i])) > 1e-5 {
				t.Fatalf("logit %d: %g != %g", i, a[i], b[i])
			}
		}
	}
	if n := testing.AllocsPerRun(10, func() {
		if err := direct.ForwardInto(ctx, b, ids, mask, nil, nil); err != nil {
			panic(err)
		}
	}); n != 0 {
		t.Fatalf("allocs %g", n)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if err := direct.ForwardInto(canceled, b, ids, mask, nil, nil); err != context.Canceled {
		t.Fatalf("cancel %v", err)
	}
	if err := direct.ForwardInto(ctx, b, ids, mask, nil, nil); err != nil {
		t.Fatal(err)
	}
}
