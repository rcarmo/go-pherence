package mosstranscribe

import (
	"fmt"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	llmmodel "github.com/rcarmo/go-pherence/model"
)

// NativeModel is the complete native MOSS-Transcribe-Diarize graph. CPU/SIMD
// execution is always available; optional GPU stages retain CPU fallback.
type NativeModel struct {
	Audio      *AudioBackbone
	Decoder    *llmmodel.LlamaModel
	GPUDecoder *llmmodel.GPUModel
	Processor  *Processor

	gpuDecoderAttempted bool
	gpuDecoderErr       error
}

func LoadNativeModel(modelDir string) (*NativeModel, error) {
	audioModel, err := LoadAudioBackbone(modelDir)
	if err != nil {
		return nil, err
	}
	decoder, err := llmmodel.LoadLlama(modelDir)
	if err != nil {
		_ = audioModel.Close()
		return nil, fmt.Errorf("MOSS decoder: %w", err)
	}
	processor, err := LoadProcessor(modelDir, audioModel.Config.AudioTokenID)
	if err != nil {
		_ = audioModel.Close()
		return nil, err
	}
	decoder.Tok = processor.Tokenizer
	return &NativeModel{Audio: audioModel, Decoder: decoder, Processor: processor}, nil
}

// EnableGPU enables all currently validated GPU stages. It returns false when
// the runtime-loaded NVIDIA backend is unavailable; CPU/SIMD remains usable.
func (m *NativeModel) EnableGPU() bool {
	if m == nil || m.Audio == nil {
		return false
	}
	audioReady := m.Audio.EnableGPU()
	if !m.gpuDecoderAttempted && m.Decoder != nil {
		m.gpuDecoderAttempted = true
		m.GPUDecoder, m.gpuDecoderErr = llmmodel.LoadGPUModel(m.Decoder)
	}
	return audioReady || m.GPUDecoder != nil
}

// GPUDecoderError reports why Qwen3 GPU residency was unavailable, if attempted.
func (m *NativeModel) GPUDecoderError() error {
	if m == nil {
		return nil
	}
	return m.gpuDecoderErr
}

// GPUEnabled reports whether at least the validated audio stage is GPU-backed.
func (m *NativeModel) GPUEnabled() bool {
	if m == nil || m.Audio == nil {
		return false
	}
	return (m.Audio.GPUEncoder != nil && m.Audio.GPUEncoder.Ready()) ||
		(m.Audio.GPUAdaptor != nil && m.Audio.GPUAdaptor.Ready()) || m.GPUDecoder != nil
}

func (m *NativeModel) Close() error {
	if m == nil {
		return nil
	}
	if m.GPUDecoder != nil {
		nvidia.SyncAll()
		m.GPUDecoder.Close()
		m.GPUDecoder = nil
	}
	if m.Audio == nil {
		return nil
	}
	return m.Audio.Close()
}

// EncodeAudio executes all independently padded chunks, retains only real-audio
// encoder rows, concatenates them in recording order, and runs merge+adaptor.
func (m *NativeModel) EncodeAudio(samples []float32) ([]float32, int, error) {
	if m == nil || m.Audio == nil || m.Audio.Encoder == nil {
		return nil, 0, fmt.Errorf("MOSS audio: nil native model")
	}
	chunks, err := ChunkAudio(samples)
	if err != nil {
		return nil, 0, err
	}
	cfg := m.Audio.Config.WhisperConfig()
	totalTokens := 0
	for _, chunk := range chunks {
		totalTokens += chunk.TokenLength
	}
	encoderRows := make([]float32, totalTokens*AudioMergeFrames*AudioWidth)
	rowOffset := 0
	for index, chunk := range chunks {
		features, err := chunk.InputFeatures(cfg)
		if err != nil {
			return nil, 0, fmt.Errorf("MOSS audio chunk %d: %w", index, err)
		}
		var hidden []float32
		if m.Audio.GPUEncoder != nil && m.Audio.GPUEncoder.Ready() {
			hidden = m.Audio.GPUEncoder.ForwardGPU(features, cfg.MaxLength)
		} else {
			hidden = m.Audio.Encoder.Forward(features, cfg.MaxLength)
		}
		rows := chunk.TokenLength * AudioMergeFrames
		values := rows * AudioWidth
		if rows > len(hidden)/AudioWidth {
			return nil, 0, fmt.Errorf("MOSS audio chunk %d: retain rows=%d encoder rows=%d", index, rows, len(hidden)/AudioWidth)
		}
		copy(encoderRows[rowOffset:rowOffset+values], hidden[:values])
		rowOffset += values
	}
	merged, tokens, ok := TimeMerge(encoderRows, totalTokens*AudioMergeFrames)
	if !ok || tokens != totalTokens {
		return nil, 0, fmt.Errorf("MOSS audio: merge tokens=%d want %d", tokens, totalTokens)
	}
	adapted := make([]float32, tokens*AdaptorHiddenDim)
	if m.Audio.GPUAdaptor == nil || !m.Audio.GPUAdaptor.Ready() || !m.Audio.GPUAdaptor.Forward(adapted, merged, tokens) {
		scratch := make([]float32, len(adapted))
		if !ForwardAdaptorTo(adapted, scratch, merged, tokens, m.Audio.Adaptor) {
			return nil, 0, fmt.Errorf("MOSS audio: adaptor failed")
		}
	}
	return adapted, tokens, nil
}

// PromptEmbeddings performs ordinary tied-embedding lookup and then exact
// masked audio insertion into a caller-independent matrix.
func (m *NativeModel) PromptEmbeddings(inputIDs []int, audioEmbeddings []float32) ([]float32, error) {
	if m == nil || m.Decoder == nil || m.Processor == nil {
		return nil, fmt.Errorf("MOSS prompt: nil native model")
	}
	hidden := m.Decoder.Config.HiddenSize
	tokenRows := make([]float32, len(inputIDs)*hidden)
	for row, tokenID := range inputIDs {
		if err := m.Decoder.TokenEmbeddingInto(tokenRows[row*hidden:(row+1)*hidden], tokenID); err != nil {
			return nil, err
		}
	}
	out := make([]float32, len(tokenRows))
	if err := InsertAudioEmbeddingsTo(out, tokenRows, audioEmbeddings, inputIDs, m.Processor.AudioTokenID, hidden); err != nil {
		return nil, err
	}
	return out, nil
}

// GenerateGreedy runs batched multimodal prefill and trims at the first EOS.
// Returned IDs contain generated tokens only, excluding the prepared prompt.
func (m *NativeModel) GenerateGreedy(inputIDs []int, audioEmbeddings []float32, maxNewTokens int) ([]int, error) {
	if maxNewTokens < 0 || maxNewTokens > GenerationMaxNewTokens {
		return nil, fmt.Errorf("MOSS generation: max_new_tokens=%d outside [0,%d]", maxNewTokens, GenerationMaxNewTokens)
	}
	embeddings, err := m.PromptEmbeddings(inputIDs, audioEmbeddings)
	if err != nil {
		return nil, err
	}
	if m.GPUDecoder != nil {
		generated, gpuErr := m.GPUDecoder.GenerateFromEmbeddingsUntil(inputIDs, embeddings, maxNewTokens, GenerationEOSTokenID)
		if gpuErr == nil {
			return generated, nil
		}
		m.gpuDecoderErr = gpuErr
	}
	all, err := m.Decoder.GenerateFromEmbeddingsUntil(inputIDs, embeddings, maxNewTokens, GenerationEOSTokenID)
	if err != nil {
		return nil, err
	}
	generated := append([]int(nil), all[len(inputIDs):]...)
	for i, token := range generated {
		if token == GenerationEOSTokenID {
			return generated[:i], nil
		}
	}
	return generated, nil
}
