package diffusiongemma

import (
	"slices"
	"testing"
)

func TestCapabilitiesReportRemainingReferenceGaps(t *testing.T) {
	caps := Capabilities()
	if caps.ReferenceComplete || caps.RuntimeReady {
		t.Fatalf("capabilities should not be globally complete yet: %+v", caps)
	}
	if caps.TextTotalOps == 0 || caps.TextImplementedOps != caps.TextTotalOps || caps.TextReferenceCompleteOps != caps.TextTotalOps {
		t.Fatalf("text op domain should be complete: %+v", caps)
	}
	if caps.VisionTotalOps == 0 || caps.VisionImplementedOps != caps.VisionTotalOps || caps.VisionReferenceCompleteOps >= caps.VisionTotalOps {
		t.Fatalf("vision op domain should be implemented but not reference-complete: %+v", caps)
	}
	if slices.Contains(caps.MissingForReference, "broader validation/default enablement for incremental prompt KV append") {
		t.Fatalf("incremental KV should no longer be reported as missing: %v", caps.MissingForReference)
	}
	if !caps.ProcessorMetadata || !caps.TokenizerMetadata || !caps.TextChatPrompt || !caps.ImageProcessorPreprocess || !caps.ImageSoftTokenPrompt || !caps.VisionTensorPlan || !caps.VisionForwardPlan || !caps.VisionTowerPrefix || !caps.VisionStreamingPrefix || caps.VisionFullStreamingMaxPatches <= 0 || !caps.VisionEmbeddingBoundary {
		t.Fatalf("processor/chat/image prompt/tensor-plan/forward-plan/tower-prefix/streaming-prefix/guarded-embedding-boundary integration should be reported complete: %+v", caps)
	}
	for _, stale := range []string{"vision/token processor integration", "vision/image processor runtime integration", "vision encoder/soft-token runtime integration", "vision tower encoder runtime integration", "vision runtime integration"} {
		if slices.Contains(caps.MissingForReference, stale) {
			t.Fatalf("%q should no longer be reported as missing: %v", stale, caps.MissingForReference)
		}
	}
	for _, want := range []string{"broader reference parity fixtures", "full image-sequence vision reference fixtures"} {
		if !slices.Contains(caps.MissingForReference, want) {
			t.Fatalf("missing %q in capabilities: %v", want, caps.MissingForReference)
		}
	}
}

func TestCapabilitiesOperationDomainFieldsMatchSummaries(t *testing.T) {
	caps := Capabilities()
	domains := OperationDomainSummaries(OperationStatuses())
	text := domains["text"]
	vision := domains["vision"]
	if caps.OperationDomains["text"] != text || caps.OperationDomains["vision"] != vision {
		t.Fatalf("capability operation_domains do not match shared summaries: caps=%+v summaries=%+v", caps.OperationDomains, domains)
	}
	if caps.TextImplementedOps != text.Implemented || caps.TextReferenceCompleteOps != text.ReferenceComplete || caps.TextTotalOps != text.Total {
		t.Fatalf("text capability fields do not match summary: caps=%+v text=%+v", caps, text)
	}
	if caps.VisionImplementedOps != vision.Implemented || caps.VisionReferenceCompleteOps != vision.ReferenceComplete || caps.VisionTotalOps != vision.Total {
		t.Fatalf("vision capability fields do not match summary: caps=%+v vision=%+v", caps, vision)
	}
	impl, ref, total := OperationStatusSummaryFromStatuses(OperationStatuses())
	if caps.ImplementedOps != impl || caps.ReferenceCompleteOps != ref || caps.TotalOps != total {
		t.Fatalf("global capability fields do not match summary: caps=%+v impl=%d ref=%d total=%d", caps, impl, ref, total)
	}
}

func TestCapabilitiesReportVisionFullStreamingPatchOverride(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_VISION_FULL_STREAMING_MAX_PATCHES", "280")
	caps := Capabilities()
	if caps.VisionFullStreamingMaxPatches != 280 || !caps.VisionFullStreamingOverride || !caps.VisionFullStreamingOverrideValid {
		t.Fatalf("vision full streaming max patches=%d override=%v override_valid=%v want 280/true/true", caps.VisionFullStreamingMaxPatches, caps.VisionFullStreamingOverride, caps.VisionFullStreamingOverrideValid)
	}
	if MaxFullStreamingVisionPatches() != 280 {
		t.Fatalf("MaxFullStreamingVisionPatches=%d want 280", MaxFullStreamingVisionPatches())
	}
}

func TestCapabilitiesReportInvalidVisionFullStreamingPatchOverride(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_VISION_FULL_STREAMING_MAX_PATCHES", "bad")
	caps := Capabilities()
	if caps.VisionFullStreamingMaxPatches != 64 || !caps.VisionFullStreamingOverride || caps.VisionFullStreamingOverrideValid {
		t.Fatalf("vision full streaming max patches=%d override=%v override_valid=%v want 64/true/false", caps.VisionFullStreamingMaxPatches, caps.VisionFullStreamingOverride, caps.VisionFullStreamingOverrideValid)
	}
}
