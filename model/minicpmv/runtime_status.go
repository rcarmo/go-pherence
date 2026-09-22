package minicpmv

func PendingRuntimeSteps() []string {
	return []string{
		"capture pinned independent MiniCPM/Qwen2/Mistral text hidden/logit parity",
		"capture independent released vision/resampler parity",
		"capture independent released MiniCPM-O audio frontend/encoder/projector parity",
		"add end-to-end MiniCPM-V/O generation parity gates",
	}
}
