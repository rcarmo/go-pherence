package vulkan

import (
	"strings"
	"testing"
	"unsafe"
)

func TestVkBufAllocRejectsMalformedSize(t *testing.T) {
	oldReady := vkReady
	vkReady = true
	t.Cleanup(func() { vkReady = oldReady })
	for _, size := range []int{0, -1} {
		if _, err := VkBufAlloc(size); err == nil || !strings.Contains(err.Error(), "invalid") {
			t.Fatalf("VkBufAlloc(%d) err=%v, want invalid size", size, err)
		}
	}
}

func TestVkBufCheckedTransferRejectsMalformedInputs(t *testing.T) {
	var nilBuf *VkBuf
	if err := nilBuf.UploadChecked([]float32{1}); err == nil || !strings.Contains(err.Error(), "not mapped") {
		t.Fatalf("nil UploadChecked err=%v", err)
	}
	if err := nilBuf.DownloadChecked([]float32{1}); err == nil || !strings.Contains(err.Error(), "not mapped") {
		t.Fatalf("nil DownloadChecked err=%v", err)
	}

	buf := &VkBuf{size: 4}
	if err := buf.UploadChecked([]float32{1}); err == nil || !strings.Contains(err.Error(), "not mapped") {
		t.Fatalf("unmapped UploadChecked err=%v", err)
	}
	storage := []byte{0, 0, 0, 0}
	buf.mapped = unsafe.Pointer(&storage[0])
	if err := buf.UploadChecked([]float32{1, 2}); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("oversized UploadChecked err=%v", err)
	}
	if err := buf.DownloadChecked([]float32{1, 2}); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("oversized DownloadChecked err=%v", err)
	}
}

func TestVkBufCheckedTransferCopiesWithinCapacity(t *testing.T) {
	storage := make([]byte, 8)
	buf := &VkBuf{size: uint64(len(storage)), mapped: unsafe.Pointer(&storage[0])}
	in := []float32{1.25, -2.5}
	if err := buf.UploadChecked(in); err != nil {
		t.Fatalf("UploadChecked: %v", err)
	}
	out := make([]float32, len(in))
	if err := buf.DownloadChecked(out); err != nil {
		t.Fatalf("DownloadChecked: %v", err)
	}
	for i := range in {
		if out[i] != in[i] {
			t.Fatalf("out[%d]=%v want %v", i, out[i], in[i])
		}
	}
}

func TestVkBufLegacyTransferNoopsOnMalformedInputs(t *testing.T) {
	var nilBuf *VkBuf
	nilBuf.Upload([]float32{1})
	nilBuf.Download([]float32{1})
	buf := &VkBuf{size: 4}
	buf.Upload([]float32{1})
	buf.Download([]float32{1})
}
