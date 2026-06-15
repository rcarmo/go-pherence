package diffusiongemma

import "testing"

func TestFullStreamingVisionPatchLimitOverrideState(t *testing.T) {
	override, valid := fullStreamingVisionPatchLimitOverrideState()
	if override || valid || MaxFullStreamingVisionPatches() != 64 {
		t.Fatalf("default override=%v valid=%v max=%d, want false/false/64", override, valid, MaxFullStreamingVisionPatches())
	}
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_VISION_FULL_STREAMING_MAX_PATCHES", "280")
	override, valid = fullStreamingVisionPatchLimitOverrideState()
	if !override || !valid || MaxFullStreamingVisionPatches() != 280 {
		t.Fatalf("valid override=%v valid=%v max=%d, want true/true/280", override, valid, MaxFullStreamingVisionPatches())
	}
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_VISION_FULL_STREAMING_MAX_PATCHES", "bad")
	override, valid = fullStreamingVisionPatchLimitOverrideState()
	if !override || valid || MaxFullStreamingVisionPatches() != 64 {
		t.Fatalf("invalid override=%v valid=%v max=%d, want true/false/64", override, valid, MaxFullStreamingVisionPatches())
	}
}

func TestBuildVisionGuardReport(t *testing.T) {
	caps := Capabilities()
	caps.VisionFullStreamingMaxPatches = 64
	guard := BuildVisionGuardReport(&ProcessorMetadata{ImageSeqLength: 280}, caps)
	if guard == nil {
		t.Fatal("nil guard report")
	}
	if guard.ProcessorPatches != 280 || guard.MaxPatches != 64 || !guard.Guarded || guard.Override || guard.OverrideValid {
		t.Fatalf("guard=%+v want processor=280 max=64 guarded=true override=false override_valid=false", guard)
	}
}

func TestBuildVisionGuardReportAllowsOverride(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_VISION_FULL_STREAMING_MAX_PATCHES", "280")
	guard := BuildVisionGuardReport(&ProcessorMetadata{ImageSeqLength: 280}, Capabilities())
	if guard == nil {
		t.Fatal("nil guard report")
	}
	if guard.ProcessorPatches != 280 || guard.MaxPatches != 280 || guard.Guarded || !guard.Override || !guard.OverrideValid {
		t.Fatalf("guard=%+v want processor=280 max=280 guarded=false override=true override_valid=true", guard)
	}
}

func TestBuildVisionGuardReportInvalidOverrideFallsBack(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_VISION_FULL_STREAMING_MAX_PATCHES", "not-a-number")
	guard := BuildVisionGuardReport(&ProcessorMetadata{ImageSeqLength: 280}, Capabilities())
	if guard == nil {
		t.Fatal("nil guard report")
	}
	if guard.MaxPatches != 64 || !guard.Guarded || !guard.Override || guard.OverrideValid {
		t.Fatalf("guard=%+v want max=64 guarded=true override=true override_valid=false", guard)
	}
}

func TestBuildVisionGuardReportNilWithoutProcessorImageSeq(t *testing.T) {
	if got := BuildVisionGuardReport(nil, Capabilities()); got != nil {
		t.Fatalf("nil processor guard=%+v want nil", got)
	}
	if got := BuildVisionGuardReport(&ProcessorMetadata{}, Capabilities()); got != nil {
		t.Fatalf("empty processor guard=%+v want nil", got)
	}
}
