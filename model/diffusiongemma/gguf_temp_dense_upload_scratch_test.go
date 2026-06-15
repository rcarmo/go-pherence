package diffusiongemma

import (
	"testing"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func TestGGUFTempDenseUploadScratchStatsAndFree(t *testing.T) {
	if !gpu.SgemmReady() {
		t.Skip("CUDA not available")
	}
	FreeGGUFTempDenseUploadScratch()
	slots, deviceBytes, hostElems := GGUFTempDenseUploadScratchStats()
	if slots != 0 || deviceBytes != 0 || hostElems != 0 {
		t.Fatalf("initial temp dense stats slots=%d device=%d host=%d, want zero", slots, deviceBytes, hostElems)
	}
	sess := beginGGUFTempDenseUploadSession()
	w := []float32{1, 2, 3, 4, 5, 6}
	buf, err := sess.Upload("test_slot", w, 2, 3)
	if err != nil {
		sess.Close()
		t.Fatal(err)
	}
	if buf == nil || buf.Ptr == 0 {
		sess.Close()
		t.Fatalf("upload returned nil/empty buffer: %+v", buf)
	}
	sess.Close()
	slots, deviceBytes, hostElems = GGUFTempDenseUploadScratchStats()
	if slots != 1 || deviceBytes < int64(len(w))*4 || hostElems < len(w) {
		t.Fatalf("after upload stats slots=%d device=%d host=%d", slots, deviceBytes, hostElems)
	}
	FreeGGUFTempDenseUploadScratch()
	slots, deviceBytes, hostElems = GGUFTempDenseUploadScratchStats()
	if slots != 0 || deviceBytes != 0 || hostElems != 0 {
		t.Fatalf("after free stats slots=%d device=%d host=%d, want zero", slots, deviceBytes, hostElems)
	}
}
