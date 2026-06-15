package diffusiongemma

import "testing"

func TestInspectContractCapabilityDomainsAndVisionGuardDefault(t *testing.T) {
	caps := Capabilities()
	if caps.OperationDomains["text"] != (OpDomainSummary{Implemented: 13, ReferenceComplete: 13, Total: 13}) {
		t.Fatalf("text operation domain=%+v", caps.OperationDomains["text"])
	}
	if caps.OperationDomains["vision"] != (OpDomainSummary{Implemented: 5, ReferenceComplete: 2, Total: 5}) {
		t.Fatalf("vision operation domain=%+v", caps.OperationDomains["vision"])
	}
	guard := BuildVisionGuardReport(&ProcessorMetadata{ImageSeqLength: 280}, caps)
	if guard == nil {
		t.Fatal("nil guard")
	}
	if guard.ProcessorPatches != 280 || guard.MaxPatches != 64 || !guard.Guarded || guard.Override || guard.OverrideValid {
		t.Fatalf("default guard=%+v", guard)
	}
}

func TestInspectContractVisionGuardValidOverride(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_VISION_FULL_STREAMING_MAX_PATCHES", "280")
	caps := Capabilities()
	guard := BuildVisionGuardReport(&ProcessorMetadata{ImageSeqLength: 280}, caps)
	if guard == nil {
		t.Fatal("nil guard")
	}
	if caps.VisionFullStreamingMaxPatches != 280 || !caps.VisionFullStreamingOverride || !caps.VisionFullStreamingOverrideValid {
		t.Fatalf("override caps max=%d override=%v valid=%v", caps.VisionFullStreamingMaxPatches, caps.VisionFullStreamingOverride, caps.VisionFullStreamingOverrideValid)
	}
	if guard.ProcessorPatches != 280 || guard.MaxPatches != 280 || guard.Guarded || !guard.Override || !guard.OverrideValid {
		t.Fatalf("valid override guard=%+v", guard)
	}
}

func TestInspectContractVisionGuardInvalidOverride(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_VISION_FULL_STREAMING_MAX_PATCHES", "invalid")
	caps := Capabilities()
	guard := BuildVisionGuardReport(&ProcessorMetadata{ImageSeqLength: 280}, caps)
	if guard == nil {
		t.Fatal("nil guard")
	}
	if caps.VisionFullStreamingMaxPatches != 64 || !caps.VisionFullStreamingOverride || caps.VisionFullStreamingOverrideValid {
		t.Fatalf("invalid override caps max=%d override=%v valid=%v", caps.VisionFullStreamingMaxPatches, caps.VisionFullStreamingOverride, caps.VisionFullStreamingOverrideValid)
	}
	if guard.ProcessorPatches != 280 || guard.MaxPatches != 64 || !guard.Guarded || !guard.Override || guard.OverrideValid {
		t.Fatalf("invalid override guard=%+v", guard)
	}
}
