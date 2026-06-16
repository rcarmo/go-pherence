package whisper

import "testing"

func TestWhisperGPUFeatureFlagsDefaultOff(t *testing.T) {
	clearWhisperGPUEnv(t)
	if UseGPULMHead() {
		t.Fatalf("LM-head GPU flag enabled by default")
	}
	if UseGPUCrossKV() {
		t.Fatalf("cross-K/V GPU flag enabled by default")
	}
	if UseGPUCrossAttention() {
		t.Fatalf("cross-attention GPU flag enabled by default")
	}
	for _, envName := range whisperGPUFeatureEnvNames() {
		if whisperGPUFeatureEnabled(envName) {
			t.Fatalf("%s enabled by default", envName)
		}
	}
}

func TestWhisperGPUFeatureFlagsUmbrella(t *testing.T) {
	clearWhisperGPUEnv(t)
	t.Setenv(envWhisperGPUGraph, "1")
	if !UseGPULMHead() {
		t.Fatalf("umbrella did not enable LM-head")
	}
	if !UseGPUCrossKV() {
		t.Fatalf("umbrella did not enable cross-K/V")
	}
	if !UseGPUCrossAttention() {
		t.Fatalf("umbrella did not enable cross-attention")
	}
	for _, envName := range whisperGPUFeatureEnvNames() {
		if !whisperGPUFeatureEnabled(envName) {
			t.Fatalf("umbrella did not enable %s", envName)
		}
	}
}

func TestWhisperGPUFeatureFlagsPerSurface(t *testing.T) {
	for _, envName := range whisperGPUFeatureEnvNames() {
		envName := envName
		t.Run(envName, func(t *testing.T) {
			clearWhisperGPUEnv(t)
			t.Setenv(envName, "1")
			if !whisperGPUFeatureEnabled(envName) {
				t.Fatalf("%s did not enable itself", envName)
			}
		})
	}
}

func whisperGPUFeatureEnvNames() []string {
	return []string{
		"GO_PHERENCE_WHISPER_GPU_MEL",
		"GO_PHERENCE_WHISPER_GPU_CONV1D",
		"GO_PHERENCE_WHISPER_GPU_ATTENTION",
		"GO_PHERENCE_WHISPER_GPU_SELF_ATTN",
		"GO_PHERENCE_WHISPER_GPU_LM_HEAD",
		"GO_PHERENCE_WHISPER_GPU_CROSS_KV",
		"GO_PHERENCE_WHISPER_GPU_CROSS_ATTN",
		"GO_PHERENCE_WHISPER_GPU_DECODER_MLP",
	}
}

func clearWhisperGPUEnv(t *testing.T) {
	t.Helper()
	t.Setenv(envWhisperGPUGraph, "")
	for _, envName := range whisperGPUFeatureEnvNames() {
		t.Setenv(envName, "")
	}
}
