package minicpmv

import "testing"

func TestCurrentCapabilities(t *testing.T) {
	caps := CurrentCapabilities()
	if !caps.ConfigParsing || !caps.MultimodalPromptPlanning || !caps.TensorShapeValidation || !caps.ValidationGate {
		t.Fatalf("expected scaffold capabilities enabled: %+v", caps)
	}
	if caps.TextRuntimeGeneration || caps.VisionTowerRuntime || caps.ResamplerRuntime || caps.AudioEncoderRuntime || caps.EndToEndGeneration {
		t.Fatalf("runtime capabilities must remain false until numeric execution lands: %+v", caps)
	}
}
