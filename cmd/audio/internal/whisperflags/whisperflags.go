// Package whisperflags centralizes Whisper GPU CLI flag handling shared by the
// audio command binaries (whisper, diarize-vtt).
package whisperflags

import "os"

// ApplyWhisperGPUCLIFlags translates the -gpu / -gpu-graph CLI flags into the
// environment-variable policy consumed before model load. -gpu-graph implies
// -gpu and enables the full GPU graph; -gpu enables the LM-head GPU policy.
func ApplyWhisperGPUCLIFlags(useGPU *bool, useGPUGraph bool) {
	if useGPUGraph {
		_ = os.Setenv("GO_PHERENCE_WHISPER_GPU_GRAPH", "1")
		if useGPU != nil {
			*useGPU = true
		}
	}
	if useGPU != nil && *useGPU {
		_ = os.Setenv("GO_PHERENCE_WHISPER_GPU_LM_HEAD", "1")
	}
}
