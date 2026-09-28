package lfm2

import (
	"fmt"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/weights"
)

// Float32TensorSource is the owned F32 tensor surface used by the LFM2 CPU
// loader. safetensors.File and safetensors.ShardedFile satisfy this contract.
type Float32TensorSource interface {
	GetFloat32(name string) ([]float32, []int, error)
}

// EmbeddingCPU owns token embeddings, final embedding norm, and an optional
// untied LM head for the correctness-first CPU path.
type EmbeddingCPU struct {
	cfg       Config
	embedding []float32
	finalNorm []float32
	lmHead    []float32
}

func LoadEmbeddingCPUFromDir(dir string, cfg Config) (*EmbeddingCPU, error) {
	src, err := weights.OpenSafetensors(dir)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	return LoadEmbeddingCPU(src, cfg)
}

// LoadEmbeddingCPU binds the embedding/final-head stage transactionally.
func LoadEmbeddingCPU(src Float32TensorSource, cfg Config) (*EmbeddingCPU, error) {
	if src == nil {
		return nil, fmt.Errorf("nil LFM2 embedding tensor source")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cfg.VocabSize == 0 {
		cfg.VocabSize = 128000
	}
	m := &EmbeddingCPU{cfg: cfg}
	var err error
	if m.embedding, err = loadLFM2Tensor(src, "model.embed_tokens.weight", []int{cfg.VocabSize, cfg.HiddenSize}); err != nil {
		return nil, err
	}
	if m.finalNorm, err = loadLFM2Tensor(src, "model.embedding_norm.weight", []int{cfg.HiddenSize}); err != nil {
		return nil, err
	}
	if cfg.TieWordEmbeddings {
		m.lmHead = m.embedding
	} else if m.lmHead, err = loadLFM2Tensor(src, "lm_head.weight", []int{cfg.VocabSize, cfg.HiddenSize}); err != nil {
		return nil, err
	}
	return m, nil
}

func loadLFM2Tensor(src Float32TensorSource, name string, want []int) ([]float32, error) {
	data, shape, err := src.GetFloat32(name)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", name, err)
	}
	if !equalLFM2Shape(shape, want) {
		return nil, fmt.Errorf("load %s: shape=%v want %v", name, shape, want)
	}
	wantLen := sizeProduct(want...)
	if wantLen < 0 || len(data) != wantLen {
		return nil, fmt.Errorf("load %s: values=%d want %d", name, len(data), wantLen)
	}
	return data, nil
}

func equalLFM2Shape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Embed copies each token row into a newly owned row-major activation matrix.
func (m *EmbeddingCPU) Embed(plan RuntimeRequestPlan, tokens []uint32) ([]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("nil LFM2 CPU embedding runtime")
	}
	contract, err := NewEmbeddingExecutionContract(m.cfg, plan)
	if err != nil {
		return nil, err
	}
	if err := contract.ValidateInput(tokens); err != nil {
		return nil, err
	}
	out := make([]float32, contract.OutputFloats)
	for i, id := range tokens {
		start := int(id) * m.cfg.HiddenSize
		copy(out[i*m.cfg.HiddenSize:(i+1)*m.cfg.HiddenSize], m.embedding[start:start+m.cfg.HiddenSize])
	}
	return out, nil
}

// FinalLogits applies the checkpoint's final RMSNorm and tied or untied LM
// head to one hidden vector.
func (m *EmbeddingCPU) FinalLogits(hidden []float32) ([]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("nil LFM2 CPU embedding runtime")
	}
	if len(hidden) != m.cfg.HiddenSize {
		return nil, fmt.Errorf("invalid LFM2 final hidden=%d want %d", len(hidden), m.cfg.HiddenSize)
	}
	normed := append([]float32(nil), hidden...)
	eps := m.cfg.NormEps
	if eps == 0 {
		eps = 1e-5
	}
	if !simd.RMSNormTo(normed, m.finalNorm, float32(eps)) {
		return nil, fmt.Errorf("LFM2 final RMSNorm failed")
	}
	logits := make([]float32, m.cfg.VocabSize)
	if !simd.GemvRows(logits, normed, m.lmHead, m.cfg.VocabSize, m.cfg.HiddenSize) {
		return nil, fmt.Errorf("LFM2 LM head failed")
	}
	return logits, nil
}
