package whisperflags

import (
	"os"
	"testing"
)

func TestApplyWhisperGPUCLIFlags(t *testing.T) {
	for _, k := range []string{"GO_PHERENCE_WHISPER_GPU_GRAPH", "GO_PHERENCE_WHISPER_GPU_LM_HEAD"} {
		t.Setenv(k, "")
	}
	useGPU := false
	ApplyWhisperGPUCLIFlags(&useGPU, false)
	if useGPU || os.Getenv("GO_PHERENCE_WHISPER_GPU_GRAPH") == "1" || os.Getenv("GO_PHERENCE_WHISPER_GPU_LM_HEAD") == "1" {
		t.Fatal("default flags should not enable GPU env")
	}
	useGPU = true
	ApplyWhisperGPUCLIFlags(&useGPU, false)
	if os.Getenv("GO_PHERENCE_WHISPER_GPU_GRAPH") == "1" {
		t.Fatal("-gpu should not enable full GPU graph")
	}
	if os.Getenv("GO_PHERENCE_WHISPER_GPU_LM_HEAD") != "1" {
		t.Fatal("-gpu should enable LM-head GPU policy before model load")
	}
	for _, k := range []string{"GO_PHERENCE_WHISPER_GPU_GRAPH", "GO_PHERENCE_WHISPER_GPU_LM_HEAD"} {
		t.Setenv(k, "")
	}
	useGPU = false
	ApplyWhisperGPUCLIFlags(&useGPU, true)
	if !useGPU || os.Getenv("GO_PHERENCE_WHISPER_GPU_GRAPH") != "1" || os.Getenv("GO_PHERENCE_WHISPER_GPU_LM_HEAD") != "1" {
		t.Fatal("-gpu-graph should imply -gpu and enable graph + LM-head policy")
	}
}
