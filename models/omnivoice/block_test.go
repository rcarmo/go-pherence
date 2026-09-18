package omnivoice

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	config "github.com/rcarmo/go-pherence/loader/omnivoice"
)

type fixtureTensor struct {
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
}
type blockFixture struct {
	Config struct {
		Dimensions struct {
			SeqLen       int `json:"seq_len"`
			Hidden       int `json:"hidden_size"`
			Intermediate int `json:"intermediate_size"`
			Heads        int `json:"num_attention_heads"`
			KVHeads      int `json:"num_key_value_heads"`
			HeadDim      int `json:"head_dim"`
		} `json:"dimensions"`
		Epsilon float64 `json:"epsilon"`
		Theta   float64 `json:"theta"`
	} `json:"config"`
	Input   fixtureTensor            `json:"input"`
	Weights map[string]fixtureTensor `json:"weights"`
	Mask    fixtureTensor            `json:"attention_mask_block_key2"`
	Outputs map[string]fixtureTensor `json:"outputs"`
}

func loadFixture(t testing.TB) (*Block, blockFixture) {
	t.Helper()
	raw, err := os.ReadFile("../../testdata/omnivoice/tiny-block.json")
	if err != nil {
		t.Fatal(err)
	}
	var f blockFixture
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	d := f.Config.Dimensions
	c := config.LLMConfig{Architectures: []string{"Qwen3ForCausalLM"}, ModelType: "qwen3", HiddenSize: d.Hidden, IntermediateSize: d.Intermediate, HeadDim: d.HeadDim, NumAttentionHeads: d.Heads, NumKeyValueHeads: d.KVHeads, NumHiddenLayers: 1, VocabSize: 32, HiddenAct: "silu", RMSNormEps: f.Config.Epsilon, RopeParameters: config.RopeParameters{RopeTheta: f.Config.Theta, RopeType: "default"}}
	weights := map[string][]float32{}
	for name, tensor := range f.Weights {
		weights[name] = tensor.Data
	}
	b, err := NewBlock(c, weights)
	if err != nil {
		t.Fatal(err)
	}
	return b, f
}
func TestBlockPyTorchParity(t *testing.T) {
	b, f := loadFixture(t)
	for _, name := range []string{"unmasked", "mask_block_key2"} {
		t.Run(name, func(t *testing.T) {
			var mask []float32
			if name != "unmasked" {
				mask = f.Mask.Data
			}
			got, err := b.Forward(f.Input.Data, 3, nil, mask)
			if err != nil {
				t.Fatal(err)
			}
			want := f.Outputs[name].Data
			maxDiff := float64(0)
			for i, v := range got {
				diff := math.Abs(float64(v - want[i]))
				maxDiff = math.Max(maxDiff, diff)
				if diff > 2e-5 || math.IsNaN(diff) {
					t.Fatalf("index %d: got %.9g want %.9g diff %g", i, v, want[i], diff)
				}
			}
			t.Logf("max absolute error: %.9g", maxDiff)
		})
	}
}
func TestBlockValidation(t *testing.T) {
	b, f := loadFixture(t)
	if _, err := b.Forward(f.Input.Data, 0, nil, nil); err == nil {
		t.Fatal("zero tokens accepted")
	}
	if _, err := b.Forward(f.Input.Data, 3, []int{1}, nil); err == nil {
		t.Fatal("bad positions accepted")
	}
	mask := make([]float32, 9)
	for i := range mask {
		mask[i] = float32(math.Inf(-1))
	}
	if _, err := b.Forward(f.Input.Data, 3, nil, mask); err == nil {
		t.Fatal("fully masked rows accepted")
	}
	if _, err := NewBlock(b.config, map[string][]float32{}); err == nil {
		t.Fatal("missing weights accepted")
	}
}
func TestBlockNonCausal(t *testing.T) {
	b, f := loadFixture(t)
	a, _ := b.Forward(f.Input.Data, 3, nil, nil)
	changed := append([]float32(nil), f.Input.Data...)
	changed[len(changed)-1] += 1
	out, _ := b.Forward(changed, 3, nil, nil)
	equal := true
	for i := 0; i < b.config.HiddenSize; i++ {
		equal = equal && out[i] == a[i]
	}
	if equal {
		t.Fatal("future token did not affect first token")
	}
}
func BenchmarkTinyBlock(b *testing.B) {
	block, f := loadFixture(b)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := block.Forward(f.Input.Data, 3, nil, nil); err != nil {
			b.Fatal(err)
		}
	}
}
