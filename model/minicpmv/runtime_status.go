package minicpmv

func PendingRuntimeSteps() []string {
	return []string{
		"capture pinned independent MiniCPM/Qwen2/Mistral text hidden/logit parity",
		"capture independent released vision/resampler parity",
		"inject MiniCPM-O audio embeddings into text backbone",
		"execute MiniCPM-O audio feature extraction and encoder",
		"add end-to-end MiniCPM-V/O generation parity gates",
	}
}
