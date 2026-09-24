// Package mojev validates a pinned metadata-only boundary for a future native
// MoJev inference port. It does not load weights or execute the scorer.
package mojev

import (
	"encoding/json"
	"fmt"
	"io"
	"reflect"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

const MaxConfigBytes = 64 << 10
const ModelRevision = "0c8695b6252f4205907433d4e196a94f032e60c3"
const SourceRevision = "a74d58cd19ec573e83e8e27f9fecd837b8d830fb"

// Layout records the admitted checkpoint metadata. Ready stays false until
// model weights, tree masking, readout and released logits qualify separately.
type Layout struct {
	Rank          int
	ContextTokens int
	Hidden        int
	TextLayers    int
	LinearLayers  int
	FullLayers    int
	Options       int
	Linear        loaderconfig.Qwen35LinearAttentionShapes
	Full          loaderconfig.Qwen35FullAttentionShapes
	RuntimeReady  bool
}

// ReadConfig accepts only the pinned 0.85B PackedScorer topology. It checks
// names/dimensions from a small config; it cannot prove tensor payload coverage.
func ReadConfig(reader io.Reader) (Layout, error) {
	var empty Layout
	if reader == nil {
		return empty, fmt.Errorf("mojev: nil config reader")
	}
	data, err := io.ReadAll(&io.LimitedReader{R: reader, N: MaxConfigBytes + 1})
	if err != nil {
		return empty, err
	}
	if len(data) > MaxConfigBytes {
		return empty, fmt.Errorf("mojev: config too large")
	}
	var c struct {
		Architectures []string `json:"architectures"`
		ModelType     string   `json:"model_type"`
		DType         string   `json:"dtype"`
		Rank          int      `json:"rank"`
		ContextTokens int      `json:"context_tokens"`
		EncoderName   string   `json:"encoder_name"`
		EncoderConfig struct {
			ModelType string `json:"model_type"`
			DType     string `json:"dtype"`
			Text      struct {
				ModelType    string   `json:"model_type"`
				Hidden       int      `json:"hidden_size"`
				Vocab        int      `json:"vocab_size"`
				Layers       int      `json:"num_hidden_layers"`
				LayerTypes   []string `json:"layer_types"`
				FullInterval int      `json:"full_attention_interval"`
				Heads        int      `json:"num_attention_heads"`
				KVHeads      int      `json:"num_key_value_heads"`
				HeadDim      int      `json:"head_dim"`
				KeyHeads     int      `json:"linear_num_key_heads"`
				ValueHeads   int      `json:"linear_num_value_heads"`
				KeyHeadDim   int      `json:"linear_key_head_dim"`
				ValueHeadDim int      `json:"linear_value_head_dim"`
				ConvKernel   int      `json:"linear_conv_kernel_dim"`
				Intermediate int      `json:"intermediate_size"`
				MTP          int      `json:"mtp_num_hidden_layers"`
				DType        string   `json:"dtype"`
			} `json:"text_config"`
			Vision struct {
				ModelType string `json:"model_type"`
				Hidden    int    `json:"hidden_size"`
				Depth     int    `json:"depth"`
				Output    int    `json:"out_hidden_size"`
			} `json:"vision_config"`
		} `json:"encoder_config"`
		Schema struct {
			Fields []struct {
				Kind    string   `json:"kind"`
				Name    string   `json:"name"`
				Options []string `json:"options"`
			} `json:"fields"`
		} `json:"schema"`
	}
	if err := json.Unmarshal(data, &c); err != nil {
		return empty, fmt.Errorf("mojev: parse config: %w", err)
	}
	t := c.EncoderConfig.Text
	if !reflect.DeepEqual(c.Architectures, []string{"PackedScorer"}) || c.ModelType != "mojev-scorer" || c.DType != "bfloat16" || c.Rank != 512 || c.ContextTokens != 16384 || c.EncoderName != "Qwen/Qwen3.5-0.8B" || c.EncoderConfig.ModelType != "qwen3_5" || c.EncoderConfig.DType != "bfloat16" || t.ModelType != "qwen3_5_text" || t.DType != "bfloat16" || t.Hidden != 1024 || t.Vocab != 248320 || t.Layers != 24 || len(t.LayerTypes) != t.Layers || t.FullInterval != 4 || t.Heads != 8 || t.KVHeads != 2 || t.HeadDim != 256 || t.KeyHeads != 16 || t.ValueHeads != 16 || t.KeyHeadDim != 128 || t.ValueHeadDim != 128 || t.ConvKernel != 4 || t.Intermediate != 3584 || t.MTP != 1 || c.EncoderConfig.Vision.ModelType != "qwen3_5_vision" || c.EncoderConfig.Vision.Hidden != 768 || c.EncoderConfig.Vision.Depth != 12 || c.EncoderConfig.Vision.Output != t.Hidden {
		return empty, fmt.Errorf("mojev: unsupported pinned scorer or encoder topology")
	}
	for i, typ := range t.LayerTypes {
		want := "linear_attention"
		if (i+1)%t.FullInterval == 0 {
			want = "full_attention"
		}
		if typ != want {
			return empty, fmt.Errorf("mojev: layer %d type %q, want %q", i, typ, want)
		}
	}
	if len(c.Schema.Fields) != 1 || c.Schema.Fields[0].Name != "answer" || c.Schema.Fields[0].Kind != "choice" || len(c.Schema.Fields[0].Options) != 64 {
		return empty, fmt.Errorf("mojev: unsupported decision schema")
	}
	for i, option := range c.Schema.Fields[0].Options {
		if option != fmt.Sprintf("slot-%d", i) {
			return empty, fmt.Errorf("mojev: option %d mismatch", i)
		}
	}
	linear, err := loaderconfig.Qwen35LinearAttentionShapesFor(t.Hidden, t.ValueHeads*t.ValueHeadDim, t.KeyHeadDim, t.ConvKernel, t.ValueHeads, t.KeyHeads)
	if err != nil {
		return empty, fmt.Errorf("mojev: linear-attention shapes: %w", err)
	}
	full, err := loaderconfig.Qwen35FullAttentionShapesFor(t.Hidden, t.Heads, t.KVHeads, t.HeadDim)
	if err != nil {
		return empty, fmt.Errorf("mojev: full-attention shapes: %w", err)
	}
	return Layout{Rank: c.Rank, ContextTokens: c.ContextTokens, Hidden: t.Hidden, TextLayers: t.Layers, LinearLayers: 18, FullLayers: 6, Options: 64, Linear: linear, Full: full, RuntimeReady: false}, nil
}
