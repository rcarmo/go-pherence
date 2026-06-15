package diffusiongemma

// VisionGuardReport describes whether the current full-streaming CPU vision
// scaffold guard allows the processor-scale image patch sequence. It is
// intended for inspect/tooling; it does not imply reference readiness.
type VisionGuardReport struct {
	ProcessorPatches int  `json:"processor_patches"`
	MaxPatches       int  `json:"max_patches"`
	Guarded          bool `json:"guarded"`
	Override         bool `json:"override"`
	OverrideValid    bool `json:"override_valid"`
}

func BuildVisionGuardReport(processor *ProcessorMetadata, caps RuntimeCapabilities) *VisionGuardReport {
	if processor == nil || processor.ImageSeqLength <= 0 {
		return nil
	}
	maxPatches := caps.VisionFullStreamingMaxPatches
	override, valid := fullStreamingVisionPatchLimitOverrideState()
	return &VisionGuardReport{ProcessorPatches: processor.ImageSeqLength, MaxPatches: maxPatches, Guarded: maxPatches > 0 && processor.ImageSeqLength > maxPatches, Override: override, OverrideValid: valid}
}
