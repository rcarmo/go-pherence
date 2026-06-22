package minicpmv

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/config"
)

type AudioFeaturePlan struct {
	SamplingRate    int  `json:"sampling_rate"`
	MelBins         int  `json:"mel_bins"`
	FeatureSize     int  `json:"feature_size"`
	DurationMS      int  `json:"duration_ms,omitempty"`
	EstimatedFrames int  `json:"estimated_frames,omitempty"`
	Ready           bool `json:"ready"`
}

func BuildAudioFeaturePlan(summary config.MiniCPMVSummary, durationMS int) (AudioFeaturePlan, error) {
	plan := AudioFeaturePlan{SamplingRate: summary.AudioSamplingRate, MelBins: summary.AudioMelBins, FeatureSize: firstPositive(summary.AudioFeatureSize, summary.AudioMelBins), DurationMS: durationMS}
	if summary.AudioModelType == "" && summary.AudioSamplingRate == 0 && summary.AudioMelBins == 0 {
		return plan, nil
	}
	if plan.SamplingRate <= 0 || plan.MelBins <= 0 || plan.FeatureSize <= 0 {
		return plan, fmt.Errorf("MiniCPM-O audio feature plan: missing sampling_rate/mel_bins/feature_size")
	}
	if durationMS > 0 {
		// Whisper-style frontends commonly produce 100 frames/s with a 10ms hop.
		plan.EstimatedFrames = ceilDiv(durationMS, 10)
	}
	plan.Ready = true
	return plan, nil
}
