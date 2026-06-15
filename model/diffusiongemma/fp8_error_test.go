package diffusiongemma

import (
	"strings"
	"testing"
)

func TestDecodeFloatRowRejectsGenericF8E4M3WithFP8PathHint(t *testing.T) {
	err := decodeFloatRowTo(make([]float32, 1), []byte{0}, "F8_E4M3")
	if err == nil {
		t.Fatal("expected F8_E4M3 generic decode error")
	}
	if !strings.Contains(err.Error(), "FP8TextWeights/GPUFP8Model") {
		t.Fatalf("error %q does not mention FP8 path", err)
	}
}
