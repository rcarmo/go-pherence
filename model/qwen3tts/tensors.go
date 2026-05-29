package qwen3tts

import (
	"sort"
	"strings"
)

var requiredTensorMarkers = map[string][]string{
	"talker":           {"talker", "text_projection", "codec_head"},
	"code_predictor":   {"code_predictor", "codec_embedding", "small_to_mtp_projection"},
	"speech_tokenizer": {"speech_tokenizer", "decoder12hz"},
}

var optionalTensorMarkers = map[string][]string{
	"speaker_encoder": {"speaker_encoder", "ecapa"},
}

type TensorCoverage struct {
	Total           int               `json:"total"`
	Talker          int               `json:"talker"`
	CodePredictor   int               `json:"code_predictor"`
	SpeechTokenizer int               `json:"speech_tokenizer"`
	SpeakerEncoder  int               `json:"speaker_encoder"`
	Other           int               `json:"other"`
	Examples        map[string]string `json:"examples,omitempty"`
	Readiness       TensorReadiness   `json:"readiness"`
}

type TensorReadiness struct {
	Ready           bool            `json:"ready"`
	PresentRequired map[string]bool `json:"present_required"`
	MissingRequired []string        `json:"missing_required,omitempty"`
	PresentOptional map[string]bool `json:"present_optional,omitempty"`
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
	cov.Readiness = InspectTensorReadiness(sorted)
	return cov
}

func InspectTensorReadiness(names []string) TensorReadiness {
	presentRequired := make(map[string]bool, len(requiredTensorMarkers))
	var missing []string
	for group, markers := range requiredTensorMarkers {
		present := anyTensorMarker(names, markers)
		presentRequired[group] = present
		if !present {
			missing = append(missing, group)
		}
	}
	sort.Strings(missing)
	presentOptional := make(map[string]bool, len(optionalTensorMarkers))
	for group, markers := range optionalTensorMarkers {
		presentOptional[group] = anyTensorMarker(names, markers)
	}
	return TensorReadiness{Ready: len(missing) == 0, PresentRequired: presentRequired, MissingRequired: missing, PresentOptional: presentOptional}
}

func anyTensorMarker(names, markers []string) bool {
	for _, name := range names {
		s := strings.ToLower(name)
		for _, marker := range markers {
			if strings.Contains(s, marker) {
				return true
			}
		}
	}
	return false
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
