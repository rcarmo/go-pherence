package diffusiongemma

// MissingReferenceGaps lists the intentionally remaining blockers for global
// reference completeness. Text GGUF Q4_K_M is complete; current gaps are broader
// reference coverage plus full image-sequence vision validation.
func MissingReferenceGaps() []string {
	return []string{"broader reference parity fixtures", "full image-sequence vision reference fixtures"}
}
