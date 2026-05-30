package lfm2

import "fmt"

// EmbeddingLayout captures token embedding and output projection sizing for
// LFM2. It makes tied-vs-untied LM-head expectations explicit before runtime
// code starts sharing or allocating these matrices.
type EmbeddingLayout struct {
	VocabSize         int  `json:"vocab_size"`
	HiddenSize        int  `json:"hidden_size"`
	TieWordEmbeddings bool `json:"tie_word_embeddings"`
	EmbeddingFloats   int  `json:"embedding_floats"`
	LMHeadFloats      int  `json:"lm_head_floats"`
	TotalUntiedFloats int  `json:"total_untied_floats"`
	OutputSharesInput bool `json:"output_shares_input"`
}

func NewEmbeddingLayout(cfg Config) (EmbeddingLayout, error) {
	if err := cfg.Validate(); err != nil {
		return EmbeddingLayout{}, err
	}
	vocab := cfg.VocabSize
	if vocab == 0 {
		vocab = 128000
	}
	layout := EmbeddingLayout{
		VocabSize:         vocab,
		HiddenSize:        cfg.HiddenSize,
		TieWordEmbeddings: cfg.TieWordEmbeddings,
		EmbeddingFloats:   vocab * cfg.HiddenSize,
		OutputSharesInput: cfg.TieWordEmbeddings,
	}
	if !cfg.TieWordEmbeddings {
		layout.LMHeadFloats = vocab * cfg.HiddenSize
	}
	layout.TotalUntiedFloats = layout.EmbeddingFloats + layout.LMHeadFloats
	return layout, layout.Validate()
}

func (l EmbeddingLayout) Validate() error {
	if l.VocabSize <= 0 || l.HiddenSize <= 0 {
		return fmt.Errorf("invalid LFM2 embedding layout dims: %+v", l)
	}
	wantEmbedding := l.VocabSize * l.HiddenSize
	if l.EmbeddingFloats != wantEmbedding {
		return fmt.Errorf("invalid LFM2 embedding floats=%d want=%d", l.EmbeddingFloats, wantEmbedding)
	}
	wantLMHead := 0
	if !l.TieWordEmbeddings {
		wantLMHead = wantEmbedding
	}
	if l.LMHeadFloats != wantLMHead {
		return fmt.Errorf("invalid LFM2 lm_head floats=%d want=%d", l.LMHeadFloats, wantLMHead)
	}
	if l.OutputSharesInput != l.TieWordEmbeddings {
		return fmt.Errorf("invalid LFM2 embedding sharing flag: shares=%v tied=%v", l.OutputSharesInput, l.TieWordEmbeddings)
	}
	if l.TotalUntiedFloats != l.EmbeddingFloats+l.LMHeadFloats {
		return fmt.Errorf("invalid LFM2 embedding total floats=%d want=%d", l.TotalUntiedFloats, l.EmbeddingFloats+l.LMHeadFloats)
	}
	return nil
}

func (l EmbeddingLayout) Bytes(bytesPerFloat int) (int64, error) {
	if err := l.Validate(); err != nil {
		return 0, err
	}
	if bytesPerFloat <= 0 {
		return 0, fmt.Errorf("invalid LFM2 embedding bytes/float=%d", bytesPerFloat)
	}
	return int64(l.TotalUntiedFloats) * int64(bytesPerFloat), nil
}
