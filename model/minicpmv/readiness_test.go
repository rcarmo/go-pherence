package minicpmv

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/config"
)

func TestBuildReadinessReport(t *testing.T) {
	runtime := RuntimePlan{ConfigReady: true, ProcessorReady: true, TokenizerReady: true, SpecialTokensReady: true, ImagePreprocessReady: true, PromptPlanningReady: true}
	tensors := &TensorReadiness{MetadataReady: true}
	shapes := TensorShapeValidation{Valid: true}
	text := TextExecutionPlan{Ready: false}
	vision := VisionExecutionPlan{Ready: false}
	audio := AudioExecutionPlan{MetadataReady: true, Ready: false}
	resampler := &ResamplerTensorPlan{Ready: true}
	report := BuildReadinessReport(config.MiniCPMVSummary{AudioModelType: "whisper_encoder"}, runtime, tensors, shapes, text, vision, audio, resampler)
	if !report.MetadataReady || !report.TensorReady || !report.ShapesReady || report.RuntimeReady {
		t.Fatalf("bad readiness flags: %+v", report)
	}
	if len(report.Blockers) < 3 {
		t.Fatalf("expected runtime blockers: %+v", report)
	}
}

func TestBuildReadinessReportMissingMetadata(t *testing.T) {
	report := BuildReadinessReport(config.MiniCPMVSummary{}, RuntimePlan{}, nil, TensorShapeValidation{}, TextExecutionPlan{}, VisionExecutionPlan{}, AudioExecutionPlan{}, nil)
	if report.MetadataReady || report.TensorReady || report.ShapesReady || report.RuntimeReady || len(report.Blockers) == 0 {
		t.Fatalf("expected blockers: %+v", report)
	}
}
