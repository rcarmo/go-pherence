package minicpmv

import (
	"sort"
	"strings"

	"github.com/rcarmo/go-pherence/loader/config"
)

type AudioOp struct {
	Name   string `json:"name"`
	Ready  bool   `json:"ready"`
	Reason string `json:"reason,omitempty"`
}

type AudioTensorRole string

const (
	AudioConv      AudioTensorRole = "conv"
	AudioAttention AudioTensorRole = "attention"
	AudioMLP       AudioTensorRole = "mlp"
	AudioNorm      AudioTensorRole = "norm"
	AudioProjector AudioTensorRole = "projector"
	AudioOther     AudioTensorRole = "other"
)

type AudioTensorBinding struct {
	Role AudioTensorRole `json:"role"`
	Name string          `json:"name"`
}

type AudioExecutionPlan struct {
	AudioModelType string                  `json:"audio_model_type,omitempty"`
	HiddenSize     int                     `json:"hidden_size,omitempty"`
	Layers         int                     `json:"layers,omitempty"`
	Heads          int                     `json:"heads,omitempty"`
	FeatureSize    int                     `json:"feature_size,omitempty"`
	MelBins        int                     `json:"mel_bins,omitempty"`
	SamplingRate   int                     `json:"sampling_rate,omitempty"`
	Bindings       []AudioTensorBinding    `json:"bindings,omitempty"`
	Counts         map[AudioTensorRole]int `json:"counts,omitempty"`
	MetadataReady  bool                    `json:"metadata_ready"`
	TensorReady    bool                    `json:"tensor_ready"`
	Ready          bool                    `json:"ready"`
	Ops            []AudioOp               `json:"ops"`
}

func BuildAudioExecutionPlan(summary config.MiniCPMVSummary, names []string) AudioExecutionPlan {
	plan := AudioExecutionPlan{
		AudioModelType: summary.AudioModelType,
		HiddenSize:     summary.AudioHiddenSize,
		Layers:         summary.AudioLayers,
		Heads:          summary.AudioHeads,
		FeatureSize:    summary.AudioFeatureSize,
		MelBins:        summary.AudioMelBins,
		SamplingRate:   summary.AudioSamplingRate,
		Counts:         map[AudioTensorRole]int{},
	}
	plan.MetadataReady = summary.AudioModelType != "" || summary.AudioHiddenSize > 0 || summary.AudioMelBins > 0 || summary.AudioSamplingRate > 0
	sorted := append([]string(nil), names...)
	sort.Strings(sorted)
	for _, name := range sorted {
		if ClassifyTensorName(name) != TensorAudioEncoder {
			continue
		}
		role := ClassifyAudioTensorName(name)
		plan.Bindings = append(plan.Bindings, AudioTensorBinding{Role: role, Name: name})
		plan.Counts[role]++
	}
	plan.TensorReady = len(plan.Bindings) > 0 && (plan.Counts[AudioConv] > 0 || plan.Counts[AudioAttention] > 0 || plan.Counts[AudioProjector] > 0)
	add := func(name string, ready bool, reason string) {
		plan.Ops = append(plan.Ops, AudioOp{Name: name, Ready: ready, Reason: reasonIf(!ready, reason)})
	}
	add("audio_metadata", plan.MetadataReady, "missing MiniCPM-O audio_config")
	add("audio_tensor_inventory", plan.TensorReady, "audio encoder tensor metadata missing")
	add("audio_feature_extraction", false, "audio feature extraction/mel frontend pending")
	add("audio_encoder_execution", false, "MiniCPM-O audio encoder tensor execution pending")
	add("audio_embedding_injection", false, "audio embedding integration pending")
	plan.Ready = false
	return plan
}

func ClassifyAudioTensorName(name string) AudioTensorRole {
	n := strings.ToLower(name)
	switch {
	case strings.Contains(n, "conv"):
		return AudioConv
	case strings.Contains(n, "self_attn") || strings.Contains(n, "attn") || strings.Contains(n, "q_proj") || strings.Contains(n, "k_proj") || strings.Contains(n, "v_proj") || strings.Contains(n, "out_proj"):
		return AudioAttention
	case strings.Contains(n, "mlp") || strings.Contains(n, "fc") || strings.Contains(n, "gate_proj") || strings.Contains(n, "up_proj") || strings.Contains(n, "down_proj"):
		return AudioMLP
	case strings.Contains(n, "norm") || strings.Contains(n, "ln"):
		return AudioNorm
	case strings.Contains(n, "projector") || strings.Contains(n, "proj"):
		return AudioProjector
	default:
		return AudioOther
	}
}
