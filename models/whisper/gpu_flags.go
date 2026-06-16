package whisper

import "os"

const envWhisperGPUGraph = "GO_PHERENCE_WHISPER_GPU_GRAPH"

func whisperGPUFeatureEnabled(envName string) bool {
	return os.Getenv(envWhisperGPUGraph) == "1" || os.Getenv(envName) == "1"
}

// UseGPUCrossAttention reports whether decoder cross-attention should try the
// GPU-resident Whisper CUDA/PTX path. It is exported so command packages can
// choose the matching DecoderState constructor without duplicating env policy.
func UseGPUCrossAttention() bool {
	return whisperGPUFeatureEnabled("GO_PHERENCE_WHISPER_GPU_CROSS_ATTN")
}
