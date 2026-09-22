package minicpmv

func PendingRuntimeSteps() []string {
	return []string{
		"capture pinned independent MiniCPM/Qwen2/Mistral text hidden/logit parity",
		"execute legacy EVA/timm vision tower and capture independent released vision parity",
		"inject MiniCPM-O audio embeddings into text backbone",
		"execute MiniCPM-O audio feature extraction and encoder",
		"add end-to-end MiniCPM-V/O generation parity gates",
	}
}
