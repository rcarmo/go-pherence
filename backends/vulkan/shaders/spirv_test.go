package shaders

import "testing"

func TestLoadSPIRVRejectsMalformedInputsBeforeRuntime(t *testing.T) {
	if _, err := LoadSPIRV(nil, 1); err == nil {
		t.Fatal("LoadSPIRV accepted nil bytecode")
	}
	if _, err := LoadSPIRV([]byte{1, 2, 3}, 1); err == nil {
		t.Fatal("LoadSPIRV accepted misaligned bytecode")
	}
	if _, err := LoadSPIRV([]byte{0, 0, 0, 0}, 0); err == nil {
		t.Fatal("LoadSPIRV accepted zero descriptor buffers")
	}
}
