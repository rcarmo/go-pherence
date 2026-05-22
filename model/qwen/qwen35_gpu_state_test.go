package qwen

import (
	"reflect"
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func sampleQwen35ForwardState() Qwen35BaseForwardState {
	return Qwen35BaseForwardState{
		Pos:   7,
		FullK: [][]float32{{1, 2, 3}, nil, {4, 5}},
		FullV: [][]float32{{6, 7, 8}, nil, {9, 10}},
		Linear: []Qwen35LinearAttentionState{
			{Conv: []float32{11, 12}, SSM: []float32{13, 14, 15}, Pos: 5},
			{Conv: nil, SSM: []float32{16}, Pos: 6},
		},
	}
}

func TestDownloadQwen35ForwardStateGPUNil(t *testing.T) {
	if _, err := DownloadQwen35ForwardStateGPU(nil); err == nil {
		t.Fatal("expected nil-state error")
	}
}

func TestUploadDownloadQwen35ForwardStateGPUParity(t *testing.T) {
	if !nvidia.SgemmReady() {
		t.Skip("NVIDIA backend not available")
	}
	want := sampleQwen35ForwardState()
	gpu, err := UploadQwen35ForwardStateGPU(want)
	if err != nil {
		t.Fatalf("upload: %v", err)
	}
	defer gpu.Free()
	if gpu.Pos != want.Pos || gpu.Bytes == 0 {
		t.Fatalf("unexpected gpu metadata pos=%d bytes=%d", gpu.Pos, gpu.Bytes)
	}
	got, err := DownloadQwen35ForwardStateGPU(gpu)
	if err != nil {
		t.Fatalf("download: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("round-trip mismatch\n got=%+v\nwant=%+v", got, want)
	}
	gpu.Free()
	if gpu.Bytes != 0 || len(gpu.FullK) != 0 || len(gpu.LinearConv) != 0 {
		t.Fatalf("free did not clear metadata: %+v", gpu)
	}
}
