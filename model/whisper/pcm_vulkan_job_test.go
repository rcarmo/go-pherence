package whisper

import (
	"context"
	"errors"
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func TestPCMVulkanHostDecoderJobAdmission(t *testing.T) {
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "1")
	t.Setenv("GO_PHERENCE_WHISPER_GPU_GRAPH", "0")
	t.Setenv("GO_PHERENCE_WHISPER_GPU_SELF_ATTN", "0")
	w := toyPCMModel()
	w.Encoder = nil
	e := &VulkanEncoder{s: &vulkanEncoderState{gate: make(chan struct{}, 1), config: w.Config, stats: VulkanEncoderStats{Frames: w.Config.MaxLength}}}
	if err := w.ValidatePCMVulkanHostDecoder(context.Background(), e); err != nil {
		t.Fatal(err)
	}
	if err := w.ValidatePCMHostOnly(); err == nil {
		t.Fatal("host route accepted missing encoder")
	}
	if err := w.ValidatePCMVulkanHostDecoder(nil, e); err == nil {
		t.Fatal("nil context")
	}
	if err := w.ValidatePCMVulkanHostDecoder(context.Background(), nil); err == nil {
		t.Fatal("nil resident")
	}
	var nilModel *Whisper
	if err := nilModel.ValidatePCMVulkanHostDecoder(context.Background(), e); err == nil {
		t.Fatal("nil model")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := w.ValidatePCMVulkanHostDecoder(ctx, e); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "0")
	if err := w.ValidatePCMVulkanHostDecoder(context.Background(), e); err == nil {
		t.Fatal("NVIDIA enabled")
	}
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "1")
	// Empty buffer values only: no NVIDIA/device initialisation in these tests.
	w.Decoder.lmHeadGPU = &nvidia.DevBuf{}
	if err := w.ValidatePCMVulkanHostDecoder(context.Background(), e); err == nil {
		t.Fatal("GPU LM head")
	}
	w.Decoder.lmHeadGPU = nil
}
