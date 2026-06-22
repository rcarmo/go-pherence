package minicpmv

import "fmt"

func ValidateSupportSummary(s SupportSummary) error {
	c := s.Capabilities
	if s.SupportVersion == "" || s.RuntimeStatus != RuntimeStatusPending || s.RuntimeRoadmapPath == "" {
		return fmt.Errorf("MiniCPM-V/O support summary version/status/roadmap invalid: version=%q status=%q roadmap=%q", s.SupportVersion, s.RuntimeStatus, s.RuntimeRoadmapPath)
	}
	if !c.ConfigParsing || !c.ProcessorMetadata || !c.TokenizerMetadata || !c.MultimodalPromptPlanning || !c.TensorShapeValidation || !c.ValidationGate {
		return fmt.Errorf("MiniCPM-V/O scaffold capabilities are incomplete: %+v", c)
	}
	if c.EndToEndGeneration || c.TextRuntimeGeneration || c.VisionTowerRuntime || c.ResamplerRuntime || c.AudioEncoderRuntime {
		return fmt.Errorf("MiniCPM-V/O runtime capabilities should remain pending until numeric execution lands: %+v", c)
	}
	if len(s.PendingRuntimeSteps) == 0 {
		return fmt.Errorf("MiniCPM-V/O pending runtime steps are empty")
	}
	return nil
}
