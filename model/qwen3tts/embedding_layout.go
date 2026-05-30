package qwen3tts

import "fmt"

// EmbeddingLayout captures the Qwen3-TTS input/output embedding and projection
// matrices that bridge text tokens, codec/control tokens, Talker hidden states,
// and CodePredictor hidden states.
type EmbeddingLayout struct {
	TalkerTextVocabSize     int `json:"talker_text_vocab_size"`
	TalkerTextHiddenSize    int `json:"talker_text_hidden_size"`
	TalkerHiddenSize        int `json:"talker_hidden_size"`
	TalkerCodecVocabSize    int `json:"talker_codec_vocab_size"`
	CodePredictorHiddenSize int `json:"code_predictor_hidden_size"`
	CodePredictorVocabSize  int `json:"code_predictor_vocab_size"`
	TextEmbeddingFloats     int `json:"text_embedding_floats"`
	TextProjectionFloats    int `json:"text_projection_floats"`
	CodecHeadFloats         int `json:"codec_head_floats"`
	CodecEmbeddingFloats    int `json:"codec_embedding_floats"`
	TotalBridgeFloats       int `json:"total_bridge_floats"`
}

func NewEmbeddingLayout(cfg ParsedConfig) (EmbeddingLayout, error) {
	if err := cfg.Validate(); err != nil {
		return EmbeddingLayout{}, err
	}
	layout := EmbeddingLayout{
		TalkerTextVocabSize:     cfg.TalkerTextVocabSize,
		TalkerTextHiddenSize:    cfg.TalkerTextHiddenSize,
		TalkerHiddenSize:        cfg.TalkerHiddenSize,
		TalkerCodecVocabSize:    cfg.TalkerVocabSize,
		CodePredictorHiddenSize: cfg.CPHiddenSize,
		CodePredictorVocabSize:  cfg.CPVocabSize,
		TextEmbeddingFloats:     cfg.TalkerTextVocabSize * cfg.TalkerTextHiddenSize,
		TextProjectionFloats:    cfg.TalkerTextHiddenSize * cfg.TalkerHiddenSize,
		CodecHeadFloats:         cfg.TalkerHiddenSize * cfg.TalkerVocabSize,
		CodecEmbeddingFloats:    cfg.CPVocabSize * cfg.CPHiddenSize,
	}
	layout.TotalBridgeFloats = layout.TextEmbeddingFloats + layout.TextProjectionFloats + layout.CodecHeadFloats + layout.CodecEmbeddingFloats
	return layout, layout.Validate()
}

func (l EmbeddingLayout) Validate() error {
	if l.TalkerTextVocabSize <= 0 || l.TalkerTextHiddenSize <= 0 || l.TalkerHiddenSize <= 0 || l.TalkerCodecVocabSize <= 0 || l.CodePredictorHiddenSize <= 0 || l.CodePredictorVocabSize <= 0 {
		return fmt.Errorf("invalid Qwen3-TTS embedding layout dims: %+v", l)
	}
	wantTextEmbedding := l.TalkerTextVocabSize * l.TalkerTextHiddenSize
	wantTextProjection := l.TalkerTextHiddenSize * l.TalkerHiddenSize
	wantCodecHead := l.TalkerHiddenSize * l.TalkerCodecVocabSize
	wantCodecEmbedding := l.CodePredictorVocabSize * l.CodePredictorHiddenSize
	if l.TextEmbeddingFloats != wantTextEmbedding {
		return fmt.Errorf("invalid Qwen3-TTS text embedding floats=%d want=%d", l.TextEmbeddingFloats, wantTextEmbedding)
	}
	if l.TextProjectionFloats != wantTextProjection {
		return fmt.Errorf("invalid Qwen3-TTS text projection floats=%d want=%d", l.TextProjectionFloats, wantTextProjection)
	}
	if l.CodecHeadFloats != wantCodecHead {
		return fmt.Errorf("invalid Qwen3-TTS codec head floats=%d want=%d", l.CodecHeadFloats, wantCodecHead)
	}
	if l.CodecEmbeddingFloats != wantCodecEmbedding {
		return fmt.Errorf("invalid Qwen3-TTS codec embedding floats=%d want=%d", l.CodecEmbeddingFloats, wantCodecEmbedding)
	}
	wantTotal := l.TextEmbeddingFloats + l.TextProjectionFloats + l.CodecHeadFloats + l.CodecEmbeddingFloats
	if l.TotalBridgeFloats != wantTotal {
		return fmt.Errorf("invalid Qwen3-TTS embedding bridge total=%d want=%d", l.TotalBridgeFloats, wantTotal)
	}
	return nil
}
