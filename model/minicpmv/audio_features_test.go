package minicpmv

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/config"
)

func TestBuildAudioFeaturePlan(t *testing.T) {
	plan, err := BuildAudioFeaturePlan(config.MiniCPMVSummary{AudioModelType: "whisper_encoder", AudioSamplingRate: 16000, AudioMelBins: 128, AudioFeatureSize: 128}, 1234)
	if err != nil {
		t.Fatalf("BuildAudioFeaturePlan: %v", err)
	}
	if !plan.Ready || plan.EstimatedFrames != 124 || plan.MelBins != 128 || plan.FeatureSize != 128 {
		t.Fatalf("bad audio feature plan: %+v", plan)
	}
}

func TestBuildAudioFeaturePlanNoAudio(t *testing.T) {
	plan, err := BuildAudioFeaturePlan(config.MiniCPMVSummary{}, 1000)
	if err != nil || plan.Ready {
		t.Fatalf("expected no-audio plan not ready without error: %+v err=%v", plan, err)
	}
}

func TestBuildAudioFeaturePlanRejectsMissingFields(t *testing.T) {
	if _, err := BuildAudioFeaturePlan(config.MiniCPMVSummary{AudioModelType: "whisper_encoder", AudioSamplingRate: 16000}, 1000); err == nil {
		t.Fatalf("expected missing mel bins error")
	}
}
