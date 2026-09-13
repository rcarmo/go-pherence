package omnivoice

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// HubertWeights contains only the semantic_model.* tensors needed for the
// native HiggsAudioV2 HuBERT semantic extractor.
type HubertWeights struct {
	Tensors    map[string][]float32
	Shapes     map[string][]int
	SampleRate int
}

// LoadHubert loads only semantic_model.* tensors from a HiggsAudioV2 tokenizer
// checkpoint rooted at path.
func LoadHubert(path string) (*HubertWeights, error) {
	raw, err := os.ReadFile(filepath.Join(path, "config.json"))
	if err != nil {
		return nil, err
	}
	var config struct {
		ModelType          string `json:"model_type"`
		SemanticSampleRate int    `json:"semantic_sample_rate"`
		SemanticModel      struct {
			ModelType               string  `json:"model_type"`
			HiddenSize              int     `json:"hidden_size"`
			IntermediateSize        int     `json:"intermediate_size"`
			NumHiddenLayers         int     `json:"num_hidden_layers"`
			NumAttentionHeads       int     `json:"num_attention_heads"`
			NumFeatExtractLayers    int     `json:"num_feat_extract_layers"`
			NumConvPosEmbeddings    int     `json:"num_conv_pos_embeddings"`
			NumConvPosEmbeddingGrou int     `json:"num_conv_pos_embedding_groups"`
			FeatExtractNorm         string  `json:"feat_extract_norm"`
			FeatExtractActivation   string  `json:"feat_extract_activation"`
			HiddenAct               string  `json:"hidden_act"`
			FeatProjLayerNorm       bool    `json:"feat_proj_layer_norm"`
			ConvBias                bool    `json:"conv_bias"`
			LayerNormEps            float64 `json:"layer_norm_eps"`
			ConvDim                 []int   `json:"conv_dim"`
			ConvKernel              []int   `json:"conv_kernel"`
			ConvStride              []int   `json:"conv_stride"`
		} `json:"semantic_model_config"`
	}
	if err = json.Unmarshal(raw, &config); err != nil {
		return nil, err
	}
	if config.ModelType != "higgs_audio_v2_tokenizer" || config.SemanticSampleRate != 16000 {
		return nil, fmt.Errorf("omnivoice: unsupported hubert config")
	}
	s := config.SemanticModel
	if s.ModelType != "hubert" || s.HiddenSize != 768 || s.IntermediateSize != 3072 || s.NumHiddenLayers != 12 || s.NumAttentionHeads != 12 || s.NumFeatExtractLayers != 7 || s.NumConvPosEmbeddings != 128 || s.NumConvPosEmbeddingGrou != 16 || s.FeatExtractNorm != "group" || s.FeatExtractActivation != "gelu" || s.HiddenAct != "gelu" || !s.FeatProjLayerNorm || s.ConvBias || s.LayerNormEps != 1e-5 {
		return nil, fmt.Errorf("omnivoice: unsupported hubert semantic config")
	}
	for i, want := range []int{512, 512, 512, 512, 512, 512, 512} {
		if i >= len(s.ConvDim) || s.ConvDim[i] != want {
			return nil, fmt.Errorf("omnivoice: unsupported hubert conv_dim")
		}
	}
	for i, want := range []int{10, 3, 3, 3, 3, 2, 2} {
		if i >= len(s.ConvKernel) || s.ConvKernel[i] != want {
			return nil, fmt.Errorf("omnivoice: unsupported hubert conv_kernel")
		}
	}
	for i, want := range []int{5, 2, 2, 2, 2, 2, 2} {
		if i >= len(s.ConvStride) || s.ConvStride[i] != want {
			return nil, fmt.Errorf("omnivoice: unsupported hubert conv_stride")
		}
	}
	f, err := safetensors.Open(filepath.Join(path, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	w := &HubertWeights{Tensors: map[string][]float32{}, Shapes: map[string][]int{}, SampleRate: config.SemanticSampleRate}
	for _, name := range f.Names() {
		if !strings.HasPrefix(name, "semantic_model.") {
			continue
		}
		data, shape, err := f.GetFloat32(name)
		if err != nil {
			return nil, err
		}
		w.Tensors[name] = data
		w.Shapes[name] = shape
	}
	if len(w.Tensors) == 0 {
		return nil, fmt.Errorf("omnivoice: empty hubert checkpoint")
	}
	return w, nil
}
