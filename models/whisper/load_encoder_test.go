package whisper

import (
	"errors"
	"fmt"
	"testing"
)

type fakeEncoderSource struct {
	tensors map[string]fakeEncoderTensor
	seen    map[string]bool
}

type fakeEncoderTensor struct {
	shape []int
	data  []float32
}

func (s *fakeEncoderSource) GetFloat32(name string) ([]float32, []int, error) {
	tensor, ok := s.tensors[name]
	if !ok {
		return nil, nil, errors.New("missing tensor")
	}
	s.seen[name] = true
	return tensor.data, tensor.shape, nil
}

func TestLoadEncoderSourceExactContract(t *testing.T) {
	cfg := Config{NumMelBins: 4, MaxLength: 12, EncoderLayers: 2, EncoderDModel: 8, EncoderHeads: 2, EncoderFFNDim: 16, HeadDim: 4}
	source := completeEncoderSource("audio", cfg)
	enc, err := LoadEncoderSource(source, "audio", cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(source.seen) != 7+cfg.EncoderLayers*15 {
		t.Fatalf("loaded %d tensors, want %d", len(source.seen), 7+cfg.EncoderLayers*15)
	}
	if len(enc.PosEmbed) != cfg.MaxLength/2*cfg.EncoderDModel || len(enc.Layers) != cfg.EncoderLayers || enc.Layers[0].KBias != nil {
		t.Fatalf("unexpected encoder contract: pos=%d layers=%d kBias=%v", len(enc.PosEmbed), len(enc.Layers), enc.Layers[0].KBias)
	}
}

func TestLoadEncoderSourceRejectsMissingAndWrongShape(t *testing.T) {
	cfg := Config{NumMelBins: 4, MaxLength: 12, EncoderLayers: 1, EncoderDModel: 8, EncoderHeads: 2, EncoderFFNDim: 16, HeadDim: 4}
	t.Run("missing", func(t *testing.T) {
		source := completeEncoderSource("audio", cfg)
		delete(source.tensors, "audio.layers.0.fc2.bias")
		if _, err := LoadEncoderSource(source, "audio", cfg); err == nil {
			t.Fatal("accepted missing tensor")
		}
	})
	t.Run("shape", func(t *testing.T) {
		source := completeEncoderSource("audio", cfg)
		tensor := source.tensors["audio.conv1.weight"]
		tensor.shape = []int{8, 4, 2}
		source.tensors["audio.conv1.weight"] = tensor
		if _, err := LoadEncoderSource(source, "audio", cfg); err == nil {
			t.Fatal("accepted incorrect shape")
		}
	})
}

func completeEncoderSource(prefix string, cfg Config) *fakeEncoderSource {
	s := &fakeEncoderSource{tensors: map[string]fakeEncoderTensor{}, seen: map[string]bool{}}
	add := func(name string, shape ...int) {
		n := 1
		for _, dim := range shape {
			n *= dim
		}
		s.tensors[prefix+"."+name] = fakeEncoderTensor{shape: shape, data: make([]float32, n)}
	}
	add("conv1.weight", cfg.EncoderDModel, cfg.NumMelBins, 3)
	add("conv1.bias", cfg.EncoderDModel)
	add("conv2.weight", cfg.EncoderDModel, cfg.EncoderDModel, 3)
	add("conv2.bias", cfg.EncoderDModel)
	add("embed_positions.weight", cfg.MaxLength/2, cfg.EncoderDModel)
	add("layer_norm.weight", cfg.EncoderDModel)
	add("layer_norm.bias", cfg.EncoderDModel)
	for i := 0; i < cfg.EncoderLayers; i++ {
		base := fmt.Sprintf("layers.%d.", i)
		add(base+"self_attn_layer_norm.weight", cfg.EncoderDModel)
		add(base+"self_attn_layer_norm.bias", cfg.EncoderDModel)
		add(base+"self_attn.q_proj.weight", cfg.EncoderDModel, cfg.EncoderDModel)
		add(base+"self_attn.q_proj.bias", cfg.EncoderDModel)
		add(base+"self_attn.k_proj.weight", cfg.EncoderDModel, cfg.EncoderDModel)
		add(base+"self_attn.v_proj.weight", cfg.EncoderDModel, cfg.EncoderDModel)
		add(base+"self_attn.v_proj.bias", cfg.EncoderDModel)
		add(base+"self_attn.out_proj.weight", cfg.EncoderDModel, cfg.EncoderDModel)
		add(base+"self_attn.out_proj.bias", cfg.EncoderDModel)
		add(base+"final_layer_norm.weight", cfg.EncoderDModel)
		add(base+"final_layer_norm.bias", cfg.EncoderDModel)
		add(base+"fc1.weight", cfg.EncoderFFNDim, cfg.EncoderDModel)
		add(base+"fc1.bias", cfg.EncoderFFNDim)
		add(base+"fc2.weight", cfg.EncoderDModel, cfg.EncoderFFNDim)
		add(base+"fc2.bias", cfg.EncoderDModel)
	}
	return s
}
