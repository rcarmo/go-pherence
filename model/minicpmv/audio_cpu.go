package minicpmv

import (
	"context"
	"fmt"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
	audioload "github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/weights"
	"github.com/rcarmo/go-pherence/model/whisper"
)

// AudioCPU composes the Transformers-compatible Whisper frontend, the shared
// CPU Whisper encoder, MiniCPM-O's two-layer ReLU projector, and temporal
// average pooling. Weights are immutable owned F32 slices; each call owns its
// activations and result. As with the shared Whisper implementation, callers
// must exclude concurrent execution.
type AudioCPU struct {
	encoder                     *whisper.Encoder
	project1, project2          visionLinear
	melBins, samplingRate       int
	encoderHidden, outputHidden int
	maxFeatureFrames, poolStep  int
}

func LoadAudioCPUFromDir(dir string, cfg config.MiniCPMVConfig) (*AudioCPU, error) {
	src, err := weights.OpenSafetensors(dir)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	return LoadAudioCPU(src, cfg)
}

// LoadAudioCPU transactionally binds the exact MiniCPM-O Whisper-medium audio
// encoder and multimodal projector. Inference dropout must be zero and only the
// exact GELU Whisper policy is admitted.
func LoadAudioCPU(src Float32TensorSource, cfg config.MiniCPMVConfig) (*AudioCPU, error) {
	if src == nil {
		return nil, fmt.Errorf("nil MiniCPM-O audio tensor source")
	}
	audio := cfg.AudioConfig
	s := cfg.MiniCPMVSummary()
	if audio == nil || audio.ModelType != "whisper" {
		return nil, fmt.Errorf("MiniCPM-O Whisper audio_config is required")
	}
	activation := audio.ActivationFunction
	if activation == "" {
		activation = "gelu"
	}
	if activation != "gelu" || audio.Dropout != 0 || audio.ActivationDropout != 0 || audio.AttentionDropout != 0 {
		return nil, fmt.Errorf("unsupported MiniCPM-O audio policy activation=%q dropout=%g activation_dropout=%g attention_dropout=%g", activation, audio.Dropout, audio.ActivationDropout, audio.AttentionDropout)
	}
	if s.AudioHiddenSize <= 0 || s.AudioIntermediateSize <= 0 || s.AudioIntermediateSize/4 != s.AudioHiddenSize || s.AudioLayers <= 0 || s.AudioHeads <= 0 || s.AudioHiddenSize%s.AudioHeads != 0 || s.AudioFeatureSize != s.AudioMelBins || s.AudioMelBins != 80 || s.AudioSamplingRate != 16000 || s.AudioMaxSourcePositions <= 0 || s.AudioPoolStep <= 0 || s.HiddenSize <= 0 {
		return nil, fmt.Errorf("invalid MiniCPM-O audio dimensions hidden=%d intermediate=%d layers=%d heads=%d feature=%d mel=%d rate=%d positions=%d pool=%d text_hidden=%d", s.AudioHiddenSize, s.AudioIntermediateSize, s.AudioLayers, s.AudioHeads, s.AudioFeatureSize, s.AudioMelBins, s.AudioSamplingRate, s.AudioMaxSourcePositions, s.AudioPoolStep, s.HiddenSize)
	}
	maxFrames, ok := checked.MulInt(s.AudioMaxSourcePositions, 2)
	if !ok {
		return nil, fmt.Errorf("MiniCPM-O audio maximum feature frames overflow")
	}
	layerNormEps := audio.LayerNormEps
	if layerNormEps == 0 {
		layerNormEps = 1e-5
	}
	if layerNormEps != 1e-5 || !finite(layerNormEps) {
		return nil, fmt.Errorf("unsupported MiniCPM-O audio layer_norm_eps=%g", layerNormEps)
	}
	encoderCfg := whisper.Config{
		NumMelBins:    s.AudioMelBins,
		MaxLength:     maxFrames,
		EncoderLayers: s.AudioLayers,
		EncoderDModel: s.AudioHiddenSize,
		EncoderHeads:  s.AudioHeads,
		EncoderFFNDim: s.AudioIntermediateSize,
		HeadDim:       s.AudioHiddenSize / s.AudioHeads,
	}
	owned := checkedAudioTensorSource{src: src}
	encoder, err := whisper.LoadEncoderSource(owned, "apm", encoderCfg)
	if err != nil {
		return nil, err
	}
	m := &AudioCPU{
		encoder: encoder, melBins: s.AudioMelBins, samplingRate: s.AudioSamplingRate,
		encoderHidden: s.AudioHiddenSize, outputHidden: s.HiddenSize,
		maxFeatureFrames: maxFrames, poolStep: s.AudioPoolStep,
	}
	if m.project1, err = loadVisionLinear(owned, "audio_projection_layer.linear1", m.encoderHidden, m.outputHidden); err != nil {
		return nil, err
	}
	if m.project2, err = loadVisionLinear(owned, "audio_projection_layer.linear2", m.outputHidden, m.outputHidden); err != nil {
		return nil, err
	}
	return m, nil
}

// Features computes exact Transformers Whisper log-mel features. PCM must be
// mono 16 kHz and already split into a single window no longer than 30 seconds;
// this method neither resamples nor pads to 30 seconds.
func (m *AudioCPU) Features(samples []float32, samplingRate int) ([]float32, int, error) {
	if m == nil {
		return nil, 0, fmt.Errorf("nil MiniCPM-O audio runtime")
	}
	if samplingRate != m.samplingRate {
		return nil, 0, fmt.Errorf("invalid MiniCPM-O audio sampling rate=%d want %d", samplingRate, m.samplingRate)
	}
	if len(samples) < m.samplingRate/100 || len(samples) > 30*m.samplingRate {
		return nil, 0, fmt.Errorf("invalid MiniCPM-O audio samples=%d want window in [%d,%d]", len(samples), m.samplingRate/100, 30*m.samplingRate)
	}
	if err := validateFiniteTextOutput("audio PCM", samples); err != nil {
		return nil, 0, err
	}
	// MiniCPMOProcessor asks WhisperFeatureExtractor for padding="max_length",
	// then retains only ceil(samples/hop) feature frames. Padding before STFT is
	// observable at the trailing boundary, so computing a short reflected window
	// and padding its features afterward would not be equivalent.
	padded := make([]float32, 30*m.samplingRate)
	copy(padded, samples)
	full, fullFrames, err := audioload.WhisperLogMel(padded, m.melBins)
	if err != nil {
		return nil, 0, err
	}
	frames := ceilDiv(len(samples), m.samplingRate/100)
	if fullFrames != m.maxFeatureFrames || frames <= 0 || frames > fullFrames || len(full) != m.melBins*fullFrames {
		return nil, 0, fmt.Errorf("invalid MiniCPM-O audio frontend output frames=%d/%d values=%d", frames, fullFrames, len(full))
	}
	features := make([]float32, m.melBins*frames)
	for mel := 0; mel < m.melBins; mel++ {
		copy(features[mel*frames:(mel+1)*frames], full[mel*fullFrames:mel*fullFrames+frames])
	}
	return features, frames, nil
}

// EncodeAudio implements AudioEncoder for precomputed mel-major features. It
// returns pooled language embeddings in row-major [tokens, textHidden] layout.
func (m *AudioCPU) EncodeAudio(features []float32, frames, featureSize int) ([]float32, error) {
	if m == nil || m.encoder == nil {
		return nil, fmt.Errorf("nil MiniCPM-O audio runtime")
	}
	if frames <= 0 || frames > m.maxFeatureFrames || featureSize != m.melBins {
		return nil, fmt.Errorf("invalid MiniCPM-O audio features frames=%d feature=%d want feature=%d frames<=%d", frames, featureSize, m.melBins, m.maxFeatureFrames)
	}
	values, ok := checked.MulInt(frames, featureSize)
	if !ok || len(features) != values {
		return nil, fmt.Errorf("invalid MiniCPM-O audio feature values=%d want %d", len(features), values)
	}
	if err := validateFiniteTextOutput("audio feature", features); err != nil {
		return nil, err
	}
	states, err := m.encoder.ForwardContext(context.Background(), features, frames)
	if err != nil {
		return nil, err
	}
	encoderFrames := (frames + 1) / 2
	encoderValues, ok := checked.MulInt(encoderFrames, m.encoderHidden)
	if !ok || len(states) != encoderValues {
		return nil, fmt.Errorf("invalid MiniCPM-O audio encoder output values=%d want %d", len(states), encoderValues)
	}
	projectedValues, ok := checked.MulInt(encoderFrames, m.outputHidden)
	if !ok {
		return nil, fmt.Errorf("MiniCPM-O audio projected shape overflows")
	}
	projected := make([]float32, projectedValues)
	intermediate := make([]float32, m.outputHidden)
	for frame := 0; frame < encoderFrames; frame++ {
		if err := m.project1.forward(intermediate, states[frame*m.encoderHidden:(frame+1)*m.encoderHidden]); err != nil {
			return nil, err
		}
		for i := range intermediate {
			if intermediate[i] < 0 {
				intermediate[i] = 0
			}
		}
		if err := m.project2.forward(projected[frame*m.outputHidden:(frame+1)*m.outputHidden], intermediate); err != nil {
			return nil, err
		}
	}
	if encoderFrames < m.poolStep {
		return nil, fmt.Errorf("MiniCPM-O audio has no complete pooling window: encoder_frames=%d pool=%d", encoderFrames, m.poolStep)
	}
	pooledFrames := (encoderFrames-m.poolStep)/m.poolStep + 1
	if pooledFrames <= 0 {
		return nil, fmt.Errorf("MiniCPM-O audio has no complete pooling window: encoder_frames=%d pool=%d", encoderFrames, m.poolStep)
	}
	pooledValues, ok := checked.MulInt(pooledFrames, m.outputHidden)
	if !ok {
		return nil, fmt.Errorf("MiniCPM-O audio pooled shape overflows")
	}
	out := make([]float32, pooledValues)
	scale := float32(1) / float32(m.poolStep)
	for frame := 0; frame < pooledFrames; frame++ {
		dst := out[frame*m.outputHidden : (frame+1)*m.outputHidden]
		for step := 0; step < m.poolStep; step++ {
			src := projected[(frame*m.poolStep+step)*m.outputHidden : (frame*m.poolStep+step+1)*m.outputHidden]
			simd.VecScaleAdd(dst, dst, src, scale)
		}
	}
	if err := validateFiniteTextOutput("audio embedding", out); err != nil {
		return nil, err
	}
	return out, nil
}

// EncodePCM composes exact feature extraction and audio encoding.
func (m *AudioCPU) EncodePCM(samples []float32, samplingRate int) ([]float32, int, error) {
	features, frames, err := m.Features(samples, samplingRate)
	if err != nil {
		return nil, 0, err
	}
	out, err := m.EncodeAudio(features, frames, m.melBins)
	if err != nil {
		return nil, 0, err
	}
	if len(out)%m.outputHidden != 0 {
		return nil, 0, fmt.Errorf("invalid MiniCPM-O audio embedding values=%d hidden=%d", len(out), m.outputHidden)
	}
	return out, len(out) / m.outputHidden, nil
}

// EncodeAndInject composes audio encoding with the existing non-aliasing token
// embedding replacement boundary. The prompt plan must match the produced
// number of audio tokens exactly.
func (m *AudioCPU) EncodeAndInject(features []float32, frames, featureSize int, tokenEmbeddings []float32, seqLen, hidden int, plan AudioPromptPlan) ([]float32, AudioEmbeddingInjection, error) {
	meta := AudioEmbeddingInjection{SequenceLength: seqLen, HiddenSize: hidden, Audios: len(plan.AudioSpans)}
	if m == nil {
		return nil, meta, fmt.Errorf("nil MiniCPM-O audio runtime")
	}
	if len(plan.AudioSpans) != 1 {
		return nil, meta, fmt.Errorf("MiniCPM-O audio runtime expects one audio span, got %d", len(plan.AudioSpans))
	}
	embeddings, err := m.EncodeAudio(features, frames, featureSize)
	if err != nil {
		return nil, meta, err
	}
	wantValues, ok := checked.MulInt(plan.PatchTokens, hidden)
	if hidden != m.outputHidden || !ok || len(embeddings) != wantValues {
		return nil, meta, fmt.Errorf("MiniCPM-O audio prompt/runtime mismatch tokens=%d/%d hidden=%d/%d", len(embeddings)/m.outputHidden, plan.PatchTokens, hidden, m.outputHidden)
	}
	return InjectAudioEmbeddings(tokenEmbeddings, seqLen, hidden, plan, embeddings)
}

type checkedAudioTensorSource struct{ src Float32TensorSource }

func (s checkedAudioTensorSource) GetFloat32(name string) ([]float32, []int, error) {
	data, shape, err := s.src.GetFloat32(name)
	if err != nil {
		return nil, nil, err
	}
	if err := validateFiniteTextOutput(name, data); err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), data...), append([]int(nil), shape...), nil
}

var _ AudioEncoder = (*AudioCPU)(nil)
