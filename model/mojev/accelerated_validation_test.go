package mojev

import (
	"math"
	"os"
	"testing"

	cfg "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/model/qwen"
	"github.com/rcarmo/go-pherence/tensor"
)

func TestAcceleratedRejectsBeforeDeviceAccess(t *testing.T) {
	data, err := os.ReadFile("testdata/mojev_config.json")
	if err != nil {
		t.Fatal(err)
	}
	meta, err := cfg.ParseQwenNativeMTPMetadata(data)
	if err != nil {
		t.Fatal(err)
	}
	meta.BF16Trajectory = false
	// A deliberately incomplete scorer. Every invalid variant must be rejected
	// before CUDA initialisation, even on a machine with no driver installed.
	meta.VocabSize = 1
	base := func() *TextScorer {
		return &TextScorer{model: &qwen.Qwen35BaseModel{Layers: make([]qwen.Qwen35BaseLayer, 24)}, meta: meta, head: &HeadWeights{}, norm: make([]float32, 1024), embedding: make([]float32, 1024), rope: make([]float32, 8*64), eps: 1e-6}
	}
	tests := []struct {
		name   string
		mutate func(*TextScorer)
	}{
		{"kind", func(s *TextScorer) { s.model.Layers[0].Kind = "bogus" }},
		{"nil_linear", func(s *TextScorer) { s.model.Layers[0].Kind = qwen.Qwen35LinearAttentionLayerKind }},
		{"short_norm", func(s *TextScorer) { s.norm = s.norm[:2] }},
		{"nan_norm", func(s *TextScorer) { s.norm[0] = float32(math.NaN()) }},
		{"embedding", func(s *TextScorer) { s.embedding = nil }},
		{"epsilon", func(s *TextScorer) { s.eps = 0 }},
		{"rope", func(s *TextScorer) { s.rope = nil }},
		{"geometry", func(s *TextScorer) { s.meta.IntermediateSize = 16 }},
		{"nil_tensor", func(s *TextScorer) {
			s.model.Layers[0] = qwen.Qwen35BaseLayer{Kind: qwen.Qwen35LinearAttentionLayerKind, Linear: &qwen.Qwen35LinearAttentionLayer{InputNorm: tensor.FromFloat32(make([]float32, 2), []int{2})}}
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := base()
			tc.mutate(s)
			if got, err := NewNVIDIATextScorer(s, 8); err == nil || got != nil {
				t.Fatal("invalid NVIDIA scorer accepted", err)
			}
			if err := validateAcceleratedTextScorer(s, 8); err == nil {
				t.Fatal("invalid scorer accepted")
			}
		})
	}
	var zero NVIDIATextScorer
	if got, err := zero.ScoreEncoded(repairedTextRow()); err == nil || got != nil {
		t.Fatal("zero scorer accepted")
	}
	if got, err := zero.ScoreText(TextRequest{}, nil, 1, 1); err == nil || got != nil {
		t.Fatal("zero text scorer accepted")
	}
	if err := zero.Close(); err != nil {
		t.Fatal(err)
	}
	if err := zero.Close(); err != nil {
		t.Fatal(err)
	}
}
