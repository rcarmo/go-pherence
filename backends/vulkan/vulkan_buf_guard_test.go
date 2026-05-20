package vulkan

import (
	"strings"
	"testing"
	"unsafe"
)

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

func TestVkBufLegacyTransferNoopsOnMalformedInputs(t *testing.T) {
	var nilBuf *VkBuf
	nilBuf.Upload([]float32{1})
	nilBuf.Download([]float32{1})
	buf := &VkBuf{size: 4}
	buf.Upload([]float32{1})
	buf.Download([]float32{1})
}
