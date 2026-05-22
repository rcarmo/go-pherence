package qwen

import (
	"math"
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func TestBF16BitRoundTrip(t *testing.T) {
	v := float32(1.5)
	if got := bf16BitsToF32(f32ToBF16Bits(v)); got != v {
		t.Fatalf("got %v want %v", got, v)
	}
}

func TestUploadDownloadQwen35ForwardStateGPUBF16(t *testing.T) {
	if !nvidia.SgemmReady() {
		t.Skip("NVIDIA backend not available")
	}
	want := sampleQwen35ForwardState()
	gpu, err := UploadQwen35ForwardStateGPUBF16(want)
	if err != nil {
		t.Fatalf("upload: %v", err)
	}
	defer gpu.Free()
	if gpu.Bytes == 0 {
		t.Fatal("expected non-zero byte accounting")
	}
	got, err := DownloadQwen35ForwardStateGPUBF16(gpu)
	if err != nil {
		t.Fatalf("download: %v", err)
	}
	if got.Pos != want.Pos || len(got.FullK) != len(want.FullK) || len(got.Linear) != len(want.Linear) {
		t.Fatalf("metadata mismatch got=%+v want=%+v", got, want)
	}
	for i, v := range want.FullK[0] {
		if math.Abs(float64(got.FullK[0][i]-bf16BitsToF32(f32ToBF16Bits(v)))) > 0 {
			t.Fatalf("value mismatch got=%v want bf16(%v)", got.FullK[0][i], v)
		}
	}
}
