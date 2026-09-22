package minicpmv

func PendingRuntimeSteps() []string {
	return []string{
		"capture pinned independent MiniCPM/Qwen2/Mistral text hidden/logit parity",
		"execute EVA02/SigLIP vision tower",
		"execute perceiver resampler and KV projection",
		"inject image/audio embeddings into text backbone",
		"execute MiniCPM-O audio feature extraction and encoder",
		"add end-to-end MiniCPM-V/O generation parity gates",
	}
}
