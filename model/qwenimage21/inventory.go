package qwenimage21

import (
	"fmt"
	"github.com/rcarmo/go-pherence/loader/gguf"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"sort"
	"strconv"
	"strings"
)

type TensorInfo struct {
	Name  string `json:"name"`
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`
}
type Inventory struct {
	Format    string   `json:"format"`
	Total     int      `json:"total"`
	Layers    int      `json:"layers"`
	Quantized int      `json:"quantized"`
	Missing   []string `json:"missing,omitempty"`
	Invalid   []string `json:"invalid,omitempty"`
	Ready     bool     `json:"ready"`
}

var globalTensors = []string{"img_in.weight", "modulation.1.weight", "norm_out.linear.weight", "proj_out.weight", "time_text_embed.timestep_embedder.linear_1.weight", "time_text_embed.timestep_embedder.linear_2.weight", "txt_in.in_layer.weight", "txt_in.out_layer.weight", "txt_in.text_norm.weight"}
var layerSuffixes = []string{"attn.norm_k.weight", "attn.norm_q.weight", "attn.to_k.weight", "attn.to_out.0.weight", "attn.to_q.weight", "attn.to_v.weight", "img_mlp.gate_up.weight", "img_mlp.out.weight"}

func InspectGGUF(path string) (Inventory, error) {
	g, e := gguf.Open(path)
	if e != nil {
		return Inventory{}, e
	}
	defer g.Close()
	items := make([]TensorInfo, len(g.Tensors))
	for i, t := range g.Tensors {
		shape := make([]int, len(t.Shape))
		for j, d := range t.Shape {
			shape[j] = int(d)
		}
		items[i] = TensorInfo{Name: t.Name, DType: t.QType.String(), Shape: shape}
	}
	return Inspect(items, "gguf"), nil
}
func InspectSafetensors(path string) (Inventory, error) {
	f, e := safetensors.Open(path)
	if e != nil {
		return Inventory{}, e
	}
	defer f.Close()
	infos := f.TensorInfos()
	items := make([]TensorInfo, 0, len(infos))
	for n, x := range infos {
		items = append(items, TensorInfo{Name: n, DType: x.DType, Shape: x.Shape})
	}
	return Inspect(items, "safetensors"), nil
}
func Inspect(items []TensorInfo, format string) Inventory {
	set := map[string]TensorInfo{}
	layers := map[int]bool{}
	inv := Inventory{Format: format, Total: len(items)}
	for _, x := range items {
		set[x.Name] = x
		if x.DType == "Q2_K" || x.DType == "Q3_K" || x.DType == "Q4_0" || x.DType == "Q4_K" || x.DType == "Q5_0" || x.DType == "Q6_K" || x.DType == "Q8_0" || x.DType == "I8" {
			inv.Quantized++
		}
		if strings.HasPrefix(x.Name, "transformer_blocks.") {
			p := strings.Split(x.Name, ".")
			if len(p) > 2 {
				if n, e := strconv.Atoi(p[1]); e == nil {
					layers[n] = true
				}
			}
		}
	}
	expected := map[string][]int{
		"img_in.weight": {64, 4096}, "modulation.1.weight": {4096, 16384}, "norm_out.linear.weight": {4096, 4096},
		"proj_out.weight": {4096, 64}, "time_text_embed.timestep_embedder.linear_1.weight": {256, 4096},
		"time_text_embed.timestep_embedder.linear_2.weight": {4096, 4096}, "txt_in.in_layer.weight": {4096, 4096},
		"txt_in.out_layer.weight": {4096, 4096}, "txt_in.text_norm.weight": {4096},
	}
	for _, n := range globalTensors {
		if x, ok := set[n]; !ok {
			inv.Missing = append(inv.Missing, n)
		} else if !shapeMatches(x.Shape, expected[n], format) {
			inv.Invalid = append(inv.Invalid, n)
		}
	}
	layerShapes := map[string][]int{"attn.norm_k.weight": {128}, "attn.norm_q.weight": {128}, "attn.to_k.weight": {4096, 4096}, "attn.to_out.0.weight": {4096, 4096}, "attn.to_q.weight": {4096, 4096}, "attn.to_v.weight": {4096, 4096}, "img_mlp.gate_up.weight": {4096, 24576}, "img_mlp.out.weight": {12288, 4096}}
	for i := 0; i < 32; i++ {
		for _, s := range layerSuffixes {
			n := fmt.Sprintf("transformer_blocks.%d.%s", i, s)
			if x, ok := set[n]; !ok {
				inv.Missing = append(inv.Missing, n)
			} else if !shapeMatches(x.Shape, layerShapes[s], format) {
				inv.Invalid = append(inv.Invalid, n)
			}
		}
	}
	inv.Layers = len(layers)
	inv.Ready = inv.Total > 0 && inv.Layers == 32 && len(inv.Missing) == 0 && len(inv.Invalid) == 0
	sort.Strings(inv.Missing)
	sort.Strings(inv.Invalid)
	return inv
}

func shapeMatches(got, want []int, format string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		j := i
		if format == "safetensors" && len(want) == 2 {
			j = 1 - i
		}
		if got[i] != want[j] {
			return false
		}
	}
	return true
}
