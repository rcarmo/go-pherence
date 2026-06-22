package minicpmv

import "github.com/rcarmo/go-pherence/loader/config"

type ReadinessReport struct {
	MetadataReady bool     `json:"metadata_ready"`
	TensorReady   bool     `json:"tensor_ready"`
	ShapesReady   bool     `json:"shapes_ready"`
	RuntimeReady  bool     `json:"runtime_ready"`
	Blockers      []string `json:"blockers,omitempty"`
}

func BuildReadinessReport(summary config.MiniCPMVSummary, runtime RuntimePlan, tensors *TensorReadiness, shapes TensorShapeValidation, text TextExecutionPlan, vision VisionExecutionPlan, audio AudioExecutionPlan, resampler *ResamplerTensorPlan) ReadinessReport {
	report := ReadinessReport{}
	report.MetadataReady = runtime.ConfigReady && runtime.ProcessorReady && runtime.TokenizerReady && runtime.SpecialTokensReady && runtime.ImagePreprocessReady && runtime.PromptPlanningReady
	if !report.MetadataReady {
		report.Blockers = append(report.Blockers, "metadata sidecars or prompt planning are incomplete")
	}
	if tensors != nil && tensors.MetadataReady {
		report.TensorReady = true
	} else {
		report.Blockers = append(report.Blockers, "safetensor inventory is incomplete")
	}
	report.ShapesReady = shapes.Valid && len(shapes.Issues) == 0
	if !report.ShapesReady {
		report.Blockers = append(report.Blockers, "safetensor shape validation is incomplete or failing")
	}
	if !text.Ready {
		report.Blockers = append(report.Blockers, "text prefill/decode runtime pending")
	}
	if !vision.Ready {
		report.Blockers = append(report.Blockers, "vision tower/resampler runtime pending")
	}
	if (summary.AudioModelType != "" || audio.MetadataReady) && !audio.Ready {
		report.Blockers = append(report.Blockers, "MiniCPM-O audio runtime pending")
	}
	if resampler == nil || !resampler.Ready {
		report.Blockers = append(report.Blockers, "resampler tensor binding or execution pending")
	}
	report.RuntimeReady = runtime.RuntimeReady && text.Ready && vision.Ready && (summary.AudioModelType == "" || audio.Ready)
	return report
}
