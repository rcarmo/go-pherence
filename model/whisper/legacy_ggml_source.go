package whisper

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/rcarmo/go-pherence/loader/safetensors"
	legacy "github.com/rcarmo/go-pherence/loader/whisperggml"
)

// legacyGGMLSource is an explicit widened-value bridge for numerical
// qualification. Metadata describes logical owned F32 tensors after decode,
// not original quantised offsets. Original storage identity stays in File.
// It neither requantises nor provides packed inference or implicit fallback.
// Source/context must outlive synchronous checked loading; close File afterwards.
type legacyGGMLSource struct {
	ctx   context.Context
	file  *legacy.File
	names map[string]string
	infos map[string]safetensors.TensorInfo
}

func newLegacyGGMLSource(ctx context.Context, file *legacy.File, cfg Config) (*legacyGGMLSource, error) {
	if ctx == nil || file == nil {
		return nil, fmt.Errorf("legacy Whisper: nil source")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	h := file.Header()
	if h.Vocab != cfg.VocabSize || h.AudioContext*2 != cfg.MaxLength || h.AudioState != cfg.EncoderDModel || h.AudioHeads != cfg.EncoderHeads || h.AudioLayers != cfg.EncoderLayers || h.TextContext != cfg.MaxDecoderLength || h.TextState != cfg.DecoderDModel || h.TextHeads != cfg.DecoderHeads || h.TextLayers != cfg.DecoderLayers || h.Mels != cfg.NumMelBins {
		return nil, fmt.Errorf("legacy Whisper: config/header mismatch")
	}
	s := &legacyGGMLSource{ctx: ctx, file: file, names: map[string]string{}, infos: map[string]safetensors.TensorInfo{}}
	offset := int64(0)
	for _, t := range file.Tensors() {
		name, err := legacyWhisperHFName(t.Name)
		if err != nil {
			return nil, err
		}
		if _, exists := s.names[name]; exists {
			return nil, fmt.Errorf("legacy Whisper: duplicate mapped name")
		}
		shape := make([]int, len(t.Shape))
		for i, d := range t.Shape {
			shape[len(shape)-1-i] = d
		}
		// The original converter stores convolution bias as [out,1].
		// HF uses rank1; no data reorder or other squeezing is performed.
		if (t.Name == "encoder.conv1.bias" || t.Name == "encoder.conv2.bias") && len(shape) == 2 && shape[1] == 1 {
			shape = shape[:1]
		}
		if offset < 0 || t.Elements < 1 || t.Elements > (int64(^uint(0)>>1)-offset)/4 {
			return nil, fmt.Errorf("legacy Whisper: logical F32 footprint overflow")
		}
		s.names[name] = t.Name
		s.infos[name] = safetensors.TensorInfo{DType: "F32", Shape: shape, DataOffsets: [2]int{int(offset), int(offset + t.Elements*4)}}
		offset += t.Elements * 4
	}
	return s, nil
}
func (s *legacyGGMLSource) TensorInfos() map[string]safetensors.TensorInfo {
	if s == nil {
		return nil
	}
	out := make(map[string]safetensors.TensorInfo, len(s.infos))
	for n, v := range s.infos {
		v.Shape = append([]int(nil), v.Shape...)
		out[n] = v
	}
	return out
}
func (s *legacyGGMLSource) GetFloat32(name string) ([]float32, []int, error) {
	if s == nil || s.file == nil {
		return nil, nil, fmt.Errorf("legacy Whisper: nil source")
	}
	n, ok := s.names[name]
	if !ok {
		return nil, nil, fmt.Errorf("legacy Whisper: unknown HF tensor")
	}
	v, err := s.file.Float32(s.ctx, n)
	if err != nil {
		return nil, nil, err
	}
	return v, append([]int(nil), s.infos[name].Shape...), nil
}

func legacyWhisperHFName(name string) (string, error) {
	for _, pair := range [][2]string{{"encoder.positional_embedding", "model.encoder.embed_positions.weight"}, {"encoder.ln_post.", "model.encoder.layer_norm."}, {"encoder.conv1.", "model.encoder.conv1."}, {"encoder.conv2.", "model.encoder.conv2."}, {"decoder.positional_embedding", "model.decoder.embed_positions.weight"}, {"decoder.token_embedding.weight", "model.decoder.embed_tokens.weight"}, {"decoder.ln.", "model.decoder.layer_norm."}} {
		if strings.HasSuffix(pair[0], ".") {
			if strings.HasPrefix(name, pair[0]) {
				suffix := strings.TrimPrefix(name, pair[0])
				if suffix == "weight" || suffix == "bias" {
					return pair[1] + suffix, nil
				}
			}
		} else if name == pair[0] {
			return pair[1], nil
		}
	}
	fields := strings.Split(name, ".")
	if len(fields) < 5 || (fields[0] != "encoder" && fields[0] != "decoder") || fields[1] != "blocks" {
		return "", fmt.Errorf("legacy Whisper: unexpected tensor %q", name)
	}
	layer, err := strconv.Atoi(fields[2])
	if err != nil || layer < 0 || layer > 63 || strconv.Itoa(layer) != fields[2] {
		return "", fmt.Errorf("legacy Whisper: layer name")
	}
	tail := strings.Join(fields[3:], ".")
	mapped := ""
	for _, pair := range [][2]string{{"attn_ln.", "self_attn_layer_norm."}, {"cross_attn_ln.", "encoder_attn_layer_norm."}, {"mlp_ln.", "final_layer_norm."}, {"mlp.0.", "fc1."}, {"mlp.2.", "fc2."}, {"attn.query.", "self_attn.q_proj."}, {"attn.key.", "self_attn.k_proj."}, {"attn.value.", "self_attn.v_proj."}, {"attn.out.", "self_attn.out_proj."}, {"cross_attn.query.", "encoder_attn.q_proj."}, {"cross_attn.key.", "encoder_attn.k_proj."}, {"cross_attn.value.", "encoder_attn.v_proj."}, {"cross_attn.out.", "encoder_attn.out_proj."}} {
		if strings.HasPrefix(tail, pair[0]) {
			suffix := strings.TrimPrefix(tail, pair[0])
			if suffix == "weight" || suffix == "bias" {
				mapped = pair[1] + suffix
				break
			}
		}
	}
	if mapped == "" {
		return "", fmt.Errorf("legacy Whisper: unexpected tensor %q", name)
	}
	return fmt.Sprintf("model.%s.layers.%d.%s", fields[0], layer, mapped), nil
}
