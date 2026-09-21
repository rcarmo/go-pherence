package whisper

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type checkedLoadSource struct {
	tensors   map[string]fakeEncoderTensor
	infos     map[string]safetensors.TensorInfo
	reads     int
	afterRead func()
	fail      error
}

func (s *checkedLoadSource) TensorInfos() map[string]safetensors.TensorInfo { return s.infos }
func (s *checkedLoadSource) GetFloat32(name string) ([]float32, []int, error) {
	s.reads++
	if s.afterRead != nil {
		s.afterRead()
	}
	if s.fail != nil {
		return nil, nil, s.fail
	}
	t, ok := s.tensors[name]
	if !ok {
		return nil, nil, fmt.Errorf("missing %s", name)
	}
	return t.data, t.shape, nil
}

// Build an independent expected layout, using the earlier encoder fixture and
// HF decoder naming/dimensions, without using speechTensorBindings.
func checkedLoadFixture(cfg Config, dtype string) *checkedLoadSource {
	s := &checkedLoadSource{tensors: completeEncoderSource("model.encoder", cfg).tensors, infos: map[string]safetensors.TensorInfo{}}
	// The checked path also supports an odd test geometry's ceil-conv output.
	s.tensors["model.encoder.embed_positions.weight"] = fakeEncoderTensor{[]int{(cfg.MaxLength + 1) / 2, cfg.EncoderDModel}, make([]float32, ((cfg.MaxLength+1)/2)*cfg.EncoderDModel)}
	add := func(name string, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		s.tensors["model.decoder."+name] = fakeEncoderTensor{shape, make([]float32, n)}
	}
	m, ff := cfg.DecoderDModel, cfg.DecoderFFNDim
	add("embed_tokens.weight", cfg.VocabSize, m)
	add("embed_positions.weight", cfg.MaxDecoderLength, m)
	add("layer_norm.weight", m)
	add("layer_norm.bias", m)
	for i := 0; i < cfg.DecoderLayers; i++ {
		p := fmt.Sprintf("layers.%d.", i)
		for _, name := range []string{"self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm"} {
			add(p+name+".weight", m)
			add(p+name+".bias", m)
		}
		for _, attn := range []string{"self_attn", "encoder_attn"} {
			for _, proj := range []string{"q_proj", "k_proj", "v_proj", "out_proj"} {
				add(p+attn+"."+proj+".weight", m, m)
				if proj != "k_proj" {
					add(p+attn+"."+proj+".bias", m)
				}
			}
		}
		add(p+"fc1.weight", ff, m)
		add(p+"fc1.bias", ff)
		add(p+"fc2.weight", m, ff)
		add(p+"fc2.bias", m)
	}
	names := make([]string, 0, len(s.tensors))
	for name := range s.tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	offset := 0
	bytes := 2
	if dtype == "F32" {
		bytes = 4
	}
	for index, name := range names {
		tensor := s.tensors[name]
		for i := range tensor.data {
			tensor.data[i] = float32(index+1) / 128
		}
		size := len(tensor.data) * bytes
		s.infos[name] = safetensors.TensorInfo{DType: dtype, Shape: append([]int(nil), tensor.shape...), DataOffsets: [2]int{offset, offset + size}}
		offset += size
	}
	return s
}

func checkedLoadConfig() Config {
	c := toyPCMModel().Config
	c.MaxLength = 3
	c.EncoderLayers = 1
	c.DecoderLayers = 1
	c.EncoderFFNDim = 4
	c.DecoderFFNDim = 6
	return c
}

func TestCheckedLoadAllTensorsAndOwnership(t *testing.T) {
	c := checkedLoadConfig()
	s := checkedLoadFixture(c, "F32")
	w, err := LoadModelSourceChecked(context.Background(), s, c)
	if err != nil {
		t.Fatal(err)
	}
	if s.reads != 11+15*c.EncoderLayers+24*c.DecoderLayers {
		t.Fatalf("read count %d", s.reads)
	}
	if err := w.validatePCMModel(); err != nil {
		t.Fatal(err)
	}
	for _, b := range speechTensorBindings(w) {
		original := s.tensors[b.name]
		if fmt.Sprint(b.shape) != fmt.Sprint(original.shape) || len(*b.dst) != len(original.data) {
			t.Fatalf("binding shape %s", b.name)
		}
		for i, v := range *b.dst {
			if v != original.data[i] {
				t.Fatalf("misbound %s[%d]", b.name, i)
			}
		}
		original.data[0] = 999
		if (*b.dst)[0] == 999 {
			t.Fatalf("borrowed weight %s", b.name)
		}
	}
	if w.Encoder.Layers[0].KBias != nil || w.Decoder.Layers[0].SelfKBias != nil || w.Decoder.Layers[0].CrossKBias != nil {
		t.Fatal("invented key bias")
	}
	if len(w.Decoder.Layers[0].FC1Weight) != c.DecoderDModel*c.DecoderFFNDim {
		t.Fatal("decoder used encoder FFN dimensions")
	}
}

func TestCheckedLoadRejectsMetadataBeforeReads(t *testing.T) {
	c := checkedLoadConfig()
	cases := []struct {
		name string
		edit func(*checkedLoadSource)
	}{
		{"missing", func(s *checkedLoadSource) { delete(s.infos, "model.decoder.layers.0.fc2.bias") }},
		{"transposed", func(s *checkedLoadSource) {
			name := "model.decoder.layers.0.fc1.weight"
			i := s.infos[name]
			i.Shape = []int{c.DecoderDModel, c.DecoderFFNDim}
			s.infos[name] = i
		}},
		{"integer", func(s *checkedLoadSource) {
			name := "model.encoder.conv1.bias"
			i := s.infos[name]
			i.DType = "I32"
			s.infos[name] = i
		}},
		{"unknown", func(s *checkedLoadSource) { s.infos["extra.weight"] = s.infos["model.encoder.conv1.bias"] }},
		{"key_bias", func(s *checkedLoadSource) {
			s.infos["model.encoder.layers.0.self_attn.k_proj.bias"] = s.infos["model.encoder.conv1.bias"]
		}},
		{"short_extent", func(s *checkedLoadSource) {
			name := "model.encoder.conv1.bias"
			i := s.infos[name]
			i.DataOffsets[1]--
			s.infos[name] = i
		}},
		{"negative", func(s *checkedLoadSource) {
			name := "model.encoder.conv1.bias"
			i := s.infos[name]
			i.DataOffsets = [2]int{-8, 0}
			s.infos[name] = i
		}},
		{"overlap", func(s *checkedLoadSource) {
			name := "model.encoder.conv1.bias"
			i := s.infos[name]
			start := s.infos["model.encoder.conv2.bias"].DataOffsets[0]
			i.DataOffsets = [2]int{start, start + 8}
			s.infos[name] = i
		}},
		{"overflow_shape", func(s *checkedLoadSource) {
			name := "model.encoder.conv1.bias"
			i := s.infos[name]
			i.Shape = []int{math.MaxInt}
			s.infos[name] = i
		}},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			s := checkedLoadFixture(c, "F32")
			tt.edit(s)
			w, err := LoadModelSourceChecked(context.Background(), s, c)
			if err == nil || w != nil || s.reads != 0 {
				t.Fatalf("accepted metadata or read before validation: %v %d", err, s.reads)
			}
		})
	}
	s := checkedLoadFixture(c, "F32")
	c.EncoderDModel = math.MaxInt
	if w, err := LoadModelSourceChecked(context.Background(), s, c); err == nil || w != nil || s.reads != 0 {
		t.Fatal("config rejected after reads")
	}
}

func TestCheckedLoadRejectsChangedDataAndCancellation(t *testing.T) {
	c := checkedLoadConfig()
	for _, kind := range []string{"short", "shape", "nan", "inf", "source_error", "cancel"} {
		t.Run(kind, func(t *testing.T) {
			s := checkedLoadFixture(c, "F32")
			name := "model.encoder.conv1.weight"
			x := s.tensors[name]
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			sentinel := errors.New("source failure")
			switch kind {
			case "short":
				x.data = x.data[:len(x.data)-1]
			case "shape":
				x.shape = []int{len(x.data)}
			case "nan":
				x.data[0] = float32(math.NaN())
			case "inf":
				x.data[0] = float32(math.Inf(-1))
			case "source_error":
				s.fail = sentinel
			case "cancel":
				s.afterRead = cancel
			}
			s.tensors[name] = x
			w, err := LoadModelSourceChecked(ctx, s, c)
			if err == nil || w != nil || s.reads != 1 {
				t.Fatalf("failed to discard bad partial model %v reads%d", err, s.reads)
			}
			if kind == "source_error" && !errors.Is(err, sentinel) {
				t.Fatal("lost source error")
			}
			if kind == "cancel" && !errors.Is(err, context.Canceled) {
				t.Fatal("lost cancellation")
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	s := checkedLoadFixture(c, "F32")
	if _, err := LoadModelSourceChecked(ctx, s, c); !errors.Is(err, context.Canceled) || s.reads != 0 {
		t.Fatal("pre-cancel read")
	}
	if _, err := LoadModelSourceChecked(context.Background(), nil, c); err == nil {
		t.Fatal("nil source")
	}
}

func TestCheckedLoadProjectionMustBeTied(t *testing.T) {
	for _, same := range []bool{true, false} {
		c := checkedLoadConfig()
		s := checkedLoadFixture(c, "F32")
		embedding := s.tensors["model.decoder.embed_tokens.weight"]
		data := append([]float32(nil), embedding.data...)
		if !same {
			data[0]++
		}
		start := 0
		for _, i := range s.infos {
			start = max(start, i.DataOffsets[1])
		}
		s.infos["proj_out.weight"] = safetensors.TensorInfo{DType: "F32", Shape: embedding.shape, DataOffsets: [2]int{start, start + len(data)*4}}
		s.tensors["proj_out.weight"] = fakeEncoderTensor{embedding.shape, data}
		w, err := LoadModelSourceChecked(context.Background(), s, c)
		if same {
			if err != nil || w == nil {
				t.Fatal(err)
			}
		} else if err == nil || w != nil || !strings.Contains(err.Error(), "untied") {
			t.Fatal("untied projection accepted", err)
		}
	}
}

func writeCheckedSafetensors(t *testing.T, s *checkedLoadSource, dtype string) string {
	t.Helper()
	size := 0
	for _, info := range s.infos {
		size = max(size, info.DataOffsets[1])
	}
	payload := make([]byte, size)
	for name, info := range s.infos {
		for i := range s.tensors[name].data {
			// Exactly 1.0 in each source dtype; genuine reader widening is exercised.
			offset := info.DataOffsets[0]
			switch dtype {
			case "F32":
				binary.LittleEndian.PutUint32(payload[offset+i*4:], math.Float32bits(1))
			case "F16":
				binary.LittleEndian.PutUint16(payload[offset+i*2:], 0x3c00)
			case "BF16":
				binary.LittleEndian.PutUint16(payload[offset+i*2:], 0x3f80)
			}
		}
	}
	header, err := json.Marshal(s.infos)
	if err != nil {
		t.Fatal(err)
	}
	for len(header)%8 != 0 {
		header = append(header, ' ')
	}
	file := make([]byte, 8+len(header)+len(payload))
	binary.LittleEndian.PutUint64(file, uint64(len(header)))
	copy(file[8:], header)
	copy(file[8+len(header):], payload)
	path := filepath.Join(t.TempDir(), "toy.safetensors")
	if err := os.WriteFile(path, file, 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestCheckedLoadSafetensorsDTypesSurviveClose(t *testing.T) {
	for _, dtype := range []string{"F32", "F16", "BF16"} {
		t.Run(dtype, func(t *testing.T) {
			c := checkedLoadConfig()
			s := checkedLoadFixture(c, dtype)
			path := writeCheckedSafetensors(t, s, dtype)
			file, err := safetensors.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer file.Close()
			w, err := LoadModelSourceChecked(context.Background(), file, c)
			if err != nil {
				t.Fatal(err)
			}
			if err := file.Close(); err != nil {
				t.Fatal(err)
			}
			for _, b := range speechTensorBindings(w) {
				for _, value := range *b.dst {
					if value != 1 {
						t.Fatal("dtype widening/ownership failed", dtype, b.name)
					}
				}
			}
		})
	}
}
