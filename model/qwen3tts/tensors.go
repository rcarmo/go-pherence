package qwen3tts

import (
	"sort"
	"strings"
)

type TensorCoverage struct {
	Total           int               `json:"total"`
	Talker          int               `json:"talker"`
	CodePredictor   int               `json:"code_predictor"`
	SpeechTokenizer int               `json:"speech_tokenizer"`
	SpeakerEncoder  int               `json:"speaker_encoder"`
	Other           int               `json:"other"`
	Examples        map[string]string `json:"examples,omitempty"`
}

func InspectTensorNames(names []string) TensorCoverage {
	cov := TensorCoverage{Total: len(names), Examples: map[string]string{}}
	sorted := append([]string(nil), names...)
	sort.Strings(sorted)
	for _, name := range sorted {
		group := TensorGroup(name)
		switch group {
		case "talker":
			cov.Talker++
		case "code_predictor":
			cov.CodePredictor++
		case "speech_tokenizer":
			cov.SpeechTokenizer++
		case "speaker_encoder":
			cov.SpeakerEncoder++
		default:
			cov.Other++
		}
		if _, ok := cov.Examples[group]; !ok {
			cov.Examples[group] = name
		}
	}
	if len(cov.Examples) == 0 {
		cov.Examples = nil
	}
	return cov
}

func TensorGroup(name string) string {
	s := strings.ToLower(name)
	switch {
	case strings.HasPrefix(s, "speech_tokenizer.") || strings.HasPrefix(s, "speech_tokenizer/") || strings.Contains(s, "decoder12hz") || strings.Contains(s, "encoder12hz"):
		return "speech_tokenizer"
	case strings.Contains(s, "speaker_encoder") || strings.HasPrefix(s, "speaker.") || strings.HasPrefix(s, "ecapa"):
		return "speaker_encoder"
	case strings.Contains(s, "code_predictor") || strings.Contains(s, "codec_embedding") || strings.Contains(s, "small_to_mtp_projection"):
		return "code_predictor"
	case strings.HasPrefix(s, "talker.") || strings.HasPrefix(s, "model.talker") || strings.Contains(s, "text_projection") || strings.Contains(s, "text_embedding") || strings.Contains(s, "codec_head"):
		return "talker"
	default:
		return "other"
	}
}
