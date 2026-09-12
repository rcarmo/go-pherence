package whisper

import (
	"fmt"
	"os"
)

// validatePCMModel rejects missing/short weights before the opt-in path can
// reach unchecked slices or the legacy loader's nil-weight fallbacks. Models
// must stay immutable after validation. This checks lengths/layout contracts;
// tensor dtype, source shape/provenance and numerical parity require loader gates.
func (w *Whisper) validatePCMModel() error { return w.validatePCMModelForEncoder(false) }

// ValidatePCMHostOnly validates the checked host model and rejects optional GPU
// buffers/features for job adapters whose lifetime contract cannot retain native
// work. NVIDIA must be disabled before process initialisation and throughout use;
// changing environment variables after device initialisation is not a teardown.
// The model, backend flags and legacy callers remain externally synchronised.
func (w *Whisper) ValidatePCMHostOnly() error {
	if err := w.validatePCMModel(); err != nil {
		return err
	}
	if os.Getenv("GO_PHERENCE_DISABLE_NVIDIA") != "1" || whisperGPUFeatureEnabled("GO_PHERENCE_WHISPER_GPU_SELF_ATTN") {
		return fmt.Errorf("host PCM jobs require NVIDIA disabled and GPU graph/self-attention flags off")
	}
	if w.Decoder.lmHeadGPU != nil {
		return fmt.Errorf("host PCM job has GPU LM-head weights")
	}
	for _, l := range w.Decoder.Layers {
		if l.gpuFC1Weight != nil || l.gpuFC2Weight != nil {
			return fmt.Errorf("host PCM job has GPU decoder weights")
		}
	}
	return nil
}

// resident=true validates the host decoder/config only. The resident encoder
// separately validates its immutable geometry/lifetime; no host encoder retained.
func (w *Whisper) validatePCMModelForEncoder(resident bool) error {
	if w == nil || (!resident && w.Encoder == nil) || w.Decoder == nil {
		return fmt.Errorf("checked PCM transcription requires a loaded model")
	}
	c := w.Config
	if err := validatePCMConfig(c); err != nil {
		return err
	}
	e, d := w.Encoder, w.Decoder
	if d.cfg != c || len(d.Layers) != c.DecoderLayers || (!resident && (e.cfg != c || len(e.Layers) != c.EncoderLayers)) {
		return fmt.Errorf("model/config mismatch")
	}
	type tensor struct {
		name     string
		data     []float32
		size     int
		optional bool
	}
	check := func(tensors ...tensor) error {
		for _, t := range tensors {
			if len(t.data) != t.size && !(t.optional && len(t.data) == 0) {
				return fmt.Errorf("invalid %s length: got %d want %d", t.name, len(t.data), t.size)
			}
		}
		return nil
	}
	m, ff := c.EncoderDModel, c.EncoderFFNDim
	if !resident {
		if len(e.PosEmbed) != ((c.MaxLength+1)/2)*m && len(e.PosEmbed) != c.MaxLength*m {
			return fmt.Errorf("invalid encoder position embedding length")
		}
		if err := check(
			tensor{"conv1.weight", e.Conv1Weight, m * c.NumMelBins * 3, false}, tensor{"conv1.bias", e.Conv1Bias, m, false},
			tensor{"conv2.weight", e.Conv2Weight, m * m * 3, false}, tensor{"conv2.bias", e.Conv2Bias, m, false},
			tensor{"encoder.layer_norm.weight", e.FinalLNWeight, m, false}, tensor{"encoder.layer_norm.bias", e.FinalLNBias, m, false},
		); err != nil {
			return err
		}
		for i, l := range e.Layers {
			if err := check(
				tensor{"attn_ln.weight", l.AttnLNWeight, m, false}, tensor{"attn_ln.bias", l.AttnLNBias, m, false},
				tensor{"q.weight", l.QWeight, m * m, false}, tensor{"q.bias", l.QBias, m, false}, tensor{"k.weight", l.KWeight, m * m, false}, tensor{"k.bias", l.KBias, m, true},
				tensor{"v.weight", l.VWeight, m * m, false}, tensor{"v.bias", l.VBias, m, false}, tensor{"o.weight", l.OWeight, m * m, false}, tensor{"o.bias", l.OBias, m, false},
				tensor{"mlp_ln.weight", l.MLPLNWeight, m, false}, tensor{"mlp_ln.bias", l.MLPLNBias, m, false},
				tensor{"fc1.weight", l.FC1Weight, ff * m, false}, tensor{"fc1.bias", l.FC1Bias, ff, false}, tensor{"fc2.weight", l.FC2Weight, m * ff, false}, tensor{"fc2.bias", l.FC2Bias, m, false},
			); err != nil {
				return fmt.Errorf("encoder layer %d: %w", i, err)
			}
		}
	}
	if err := check(
		tensor{"decoder.embed_tokens", d.TokenEmbed, c.VocabSize * m, false}, tensor{"decoder.embed_positions", d.PosEmbed, c.MaxDecoderLength * m, false},
		tensor{"decoder.layer_norm.weight", d.FinalLNWeight, m, false}, tensor{"decoder.layer_norm.bias", d.FinalLNBias, m, false},
	); err != nil {
		return err
	}
	ff = c.DecoderFFNDim
	for i, l := range d.Layers {
		if err := check(
			tensor{"self_ln.weight", l.SelfAttnLNWeight, m, false}, tensor{"self_ln.bias", l.SelfAttnLNBias, m, false},
			tensor{"self_q.weight", l.SelfQWeight, m * m, false}, tensor{"self_q.bias", l.SelfQBias, m, false}, tensor{"self_k.weight", l.SelfKWeight, m * m, false}, tensor{"self_k.bias", l.SelfKBias, m, true},
			tensor{"self_v.weight", l.SelfVWeight, m * m, false}, tensor{"self_v.bias", l.SelfVBias, m, false}, tensor{"self_o.weight", l.SelfOWeight, m * m, false}, tensor{"self_o.bias", l.SelfOBias, m, false},
			tensor{"cross_ln.weight", l.CrossAttnLNWeight, m, false}, tensor{"cross_ln.bias", l.CrossAttnLNBias, m, false},
			tensor{"cross_q.weight", l.CrossQWeight, m * m, false}, tensor{"cross_q.bias", l.CrossQBias, m, false}, tensor{"cross_k.weight", l.CrossKWeight, m * m, false}, tensor{"cross_k.bias", l.CrossKBias, m, true},
			tensor{"cross_v.weight", l.CrossVWeight, m * m, false}, tensor{"cross_v.bias", l.CrossVBias, m, false}, tensor{"cross_o.weight", l.CrossOWeight, m * m, false}, tensor{"cross_o.bias", l.CrossOBias, m, false},
			tensor{"mlp_ln.weight", l.MLPLNWeight, m, false}, tensor{"mlp_ln.bias", l.MLPLNBias, m, false},
			tensor{"fc1.weight", l.FC1Weight, ff * m, false}, tensor{"fc1.bias", l.FC1Bias, ff, false}, tensor{"fc2.weight", l.FC2Weight, m * ff, false}, tensor{"fc2.bias", l.FC2Bias, m, false},
		); err != nil {
			return fmt.Errorf("decoder layer %d: %w", i, err)
		}
	}
	return nil
}

// Bounds are checked before products/divisions or allocation. Zero transformer
// layers are allowed for wiring fixtures; production callers use named configs.
func validatePCMConfig(c Config) error {
	if (c.NumMelBins != 80 && c.NumMelBins != 128) || c.MaxLength < 1 || c.MaxLength > 3000 || c.MaxDecoderLength < 4 || c.MaxDecoderLength > 448 || c.EncoderDModel < 1 || c.EncoderDModel > 1280 || c.DecoderDModel != c.EncoderDModel || c.EncoderFFNDim < 1 || c.EncoderFFNDim > 5120 || c.DecoderFFNDim < 1 || c.DecoderFFNDim > 5120 || c.EncoderLayers < 0 || c.EncoderLayers > 32 || c.DecoderLayers < 0 || c.DecoderLayers > 32 || c.EncoderHeads < 1 || c.EncoderHeads > c.EncoderDModel || c.DecoderHeads < 1 || c.DecoderHeads > c.DecoderDModel || c.EncoderDModel%c.EncoderHeads != 0 || c.DecoderDModel%c.DecoderHeads != 0 || c.HeadDim != c.EncoderDModel/c.EncoderHeads || c.HeadDim != c.DecoderDModel/c.DecoderHeads || (c.VocabSize != 51865 && c.VocabSize != 51866) {
		return fmt.Errorf("unsupported checked PCM model geometry")
	}
	return nil
}
