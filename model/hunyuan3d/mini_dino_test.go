package hunyuan3d

import (
	"encoding/json"
	"os"
	"reflect"
	"slices"
	"strconv"
	"testing"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type miniDINOHeaderFixture struct {
	Schema                  int    `json:"schema"`
	Repository              string `json:"repository"`
	Revision                string `json:"revision"`
	Subfolder               string `json:"subfolder"`
	ConfigSHA256            string `json:"config_sha256"`
	SafetensorsHeaderSHA256 string `json:"safetensors_header_sha256"`
	ConditionerCount        int    `json:"conditioner_count"`
	Layers                  int    `json:"layers"`
	Tensors                 map[string]struct {
		DType string `json:"dtype"`
		Shape []int  `json:"shape"`
	} `json:"tensors"`
}

func miniDINOTestModel(t *testing.T) (*Model, miniDINOHeaderFixture) {
	t.Helper()
	data, err := os.ReadFile("testdata/mini_dino_header.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture miniDINOHeaderFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.Repository != "tencent/Hunyuan3D-2mini" || fixture.Revision != "f90a0f7df7d5e6f71109cf333f6a95a0ae3194a6" || fixture.ConfigSHA256 != "cabcba7f6115752c8fe5b370e12bf714936f70377a8a80f151872f76c2d64609" || fixture.SafetensorsHeaderSHA256 != "dd8b61f43325eb7f58717bcfe0e4fafc3932cfef3bf4dee96272d52a757427b4" || fixture.Subfolder != "hunyuan3d-dit-v2-mini" {
		t.Fatal("unexpected Hunyuan3D-2mini header fixture provenance")
	}
	cfg, err := loaderconfig.ParseHunyuan3DConfig([]byte(sampleConfig))
	if err != nil {
		t.Fatal(err)
	}
	cfg.Conditioner.Params.MainImageEncoder.Kwargs = map[string]any{
		"image_size": 518,
		"config": map[string]any{
			"hidden_size": miniDINOWidth, "num_hidden_layers": miniDINOLayers,
			"num_attention_heads": 24, "num_channels": 3, "patch_size": 14,
			"image_size": 518, "use_swiglu_ffn": true, "qkv_bias": true,
		},
	}
	shape, err := FromLoaderConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	condition, err := ConditionerFromShapeConfig(shape)
	if err != nil {
		t.Fatal(err)
	}
	file := &safetensors.File{} // Metadata binding never reads tensor payloads.
	m := &Model{Config: cfg, Shape: shape, Condition: condition, checkpoint: file, Tensors: TensorGroups{Conditioner: make(map[string]TensorRef, fixture.ConditionerCount)}}
	for name, spec := range fixture.Tensors {
		if len(name) >= len("encoder.layer.0.") && name[:len("encoder.layer.0.")] == "encoder.layer.0." {
			for layer := 0; layer < fixture.Layers; layer++ {
				full := miniDINOPrefix + "encoder.layer." + strconv.Itoa(layer) + name[len("encoder.layer.0"):]
				m.Tensors.Conditioner[full] = TensorRef{Name: full, DType: spec.DType, Shape: slices.Clone(spec.Shape), file: file}
			}
			continue
		}
		full := miniDINOPrefix + name
		m.Tensors.Conditioner[full] = TensorRef{Name: full, DType: spec.DType, Shape: slices.Clone(spec.Shape), file: file}
	}
	return m, fixture
}

func TestBindMiniDINOConditionerHeader(t *testing.T) {
	m, fixture := miniDINOTestModel(t)
	if fixture.Layers != miniDINOLayers || len(fixture.Tensors) != len(miniDINOGlobalSpecs)+len(miniDINOLayerSpecs) || len(m.Tensors.Conditioner) != fixture.ConditionerCount {
		t.Fatalf("header fixture count/layers mismatch: %+v", fixture)
	}
	bound, err := m.BindMiniDINOConditioner()
	if err != nil {
		t.Fatal(err)
	}
	if len(bound.Tensors) != fixture.ConditionerCount {
		t.Fatalf("bound tensors=%d want=%d", len(bound.Tensors), fixture.ConditionerCount)
	}
	before := slices.Clone(bound.Tensors[0].Shape)
	m.Tensors.Conditioner[bound.Tensors[0].Name].Shape[0]++
	if !slices.Equal(bound.Tensors[0].Shape, before) {
		t.Fatal("bound shape aliases mutable manifest metadata")
	}
}

func TestBindMiniDINOConditionerRejectsIncompatibleHeader(t *testing.T) {
	cases := map[string]func(*Model){
		"missing tensor": func(m *Model) { delete(m.Tensors.Conditioner, miniDINOPrefix+"encoder.layer.39.norm2.weight") },
		"extra tensor":   func(m *Model) { m.Tensors.Conditioner[miniDINOPrefix+"unexpected"] = TensorRef{} },
		"wrong dtype": func(m *Model) {
			ref := m.Tensors.Conditioner[miniDINOPrefix+"embeddings.cls_token"]
			ref.DType = "F32"
			m.Tensors.Conditioner[ref.Name] = ref
		},
		"wrong shape": func(m *Model) {
			ref := m.Tensors.Conditioner[miniDINOPrefix+"encoder.layer.20.mlp.weights_in.weight"]
			ref.Shape[0]--
			m.Tensors.Conditioner[ref.Name] = ref
		},
		"wrong owner": func(m *Model) {
			ref := m.Tensors.Conditioner[miniDINOPrefix+"layernorm.bias"]
			ref.file = &safetensors.File{}
			m.Tensors.Conditioner[ref.Name] = ref
		},
		"wrong encoder":       func(m *Model) { m.Config.Conditioner.Params.MainImageEncoder.Type = "CLIPImageEncoder" },
		"additional encoder":  func(m *Model) { m.Config.Conditioner.Params.AdditionalImageEncoder.Type = "CLIPImageEncoder" },
		"wrong context width": func(m *Model) { m.Shape.ContextInDim++ },
		"wrong image size":    func(m *Model) { m.Condition.ImageSize = 512 },
		"wrong layers": func(m *Model) {
			m.Config.Conditioner.Params.MainImageEncoder.Kwargs["config"].(map[string]any)["num_hidden_layers"] = 39
		},
		"wrong heads": func(m *Model) {
			m.Config.Conditioner.Params.MainImageEncoder.Kwargs["config"].(map[string]any)["num_attention_heads"] = 16
		},
		"wrong patch": func(m *Model) {
			m.Config.Conditioner.Params.MainImageEncoder.Kwargs["config"].(map[string]any)["patch_size"] = 16
		},
		"missing config": func(m *Model) { delete(m.Config.Conditioner.Params.MainImageEncoder.Kwargs, "config") },
		"missing kwargs": func(m *Model) { m.Config.Conditioner.Params.MainImageEncoder.Kwargs = nil },
		"ungated MLP": func(m *Model) {
			m.Config.Conditioner.Params.MainImageEncoder.Kwargs["config"].(map[string]any)["use_swiglu_ffn"] = false
		},
		"closed model": func(m *Model) { m.checkpoint = nil },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			m, _ := miniDINOTestModel(t)
			mutate(m)
			before := len(m.Tensors.Conditioner)
			bound, err := m.BindMiniDINOConditioner()
			if err == nil || bound != nil || len(m.Tensors.Conditioner) != before {
				t.Fatalf("accepted or mutated invalid header: err=%v", err)
			}
		})
	}
	if got, err := (*Model)(nil).BindMiniDINOConditioner(); err == nil || !reflect.DeepEqual(got, (*MiniDINOConditioner)(nil)) {
		t.Fatal("nil model accepted")
	}
}
