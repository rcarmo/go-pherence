package lmhead

import "testing"

func TestChunkedGPULMHeadRejectsNilModel(t *testing.T) {
	if (*GPUModel)(nil).chunkedGPULMHead(make([]float32, 1), []float32{1}, 1, 1) {
		t.Fatal("accepted nil GPUModel")
	}
}
