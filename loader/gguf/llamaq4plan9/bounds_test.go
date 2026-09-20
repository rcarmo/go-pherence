package llamaq4plan9

import "testing"

// Malformed calls must return before native entry even when ISA is available.
// These tests also run against unsupported stubs, without inventing a fallback.
func TestMalformedTilesNeverEnterNative(t *testing.T) {
	q4, q8 := make([]byte, 144), make([]byte, 544)
	if err := DotQ4_0x8Q8_0x4CompilerPlan9(q4, q8[:136], 1, nil); err == nil {
		t.Fatal("nil output")
	}
	if err := DotQ4_0x8Q8_0x8CompilerPlan9(q4, q8[:272], 1, nil); err == nil {
		t.Fatal("nil output")
	}
	if err := DotQ4_0x8Q8_0x8StagePlan9(q4, q8[:272], 1, nil); err == nil {
		t.Fatal("nil output")
	}
	if err := DotQ4_0x8Q8_0x16CompilerPlan9(q4, q8, 1, nil); err == nil {
		t.Fatal("nil output")
	}
	if err := DotQ4_0x8Q8_0x16PairPlan9(q4, q8, 1, nil); err == nil {
		t.Fatal("nil output")
	}
	var out [32]float32
	if err := DotQ4_0x8Q8_0x4CompilerPlan9(nil, nil, int(^uint(0)>>1)/8+1, &out); err == nil {
		t.Fatal("wrapped buffers")
	}
}
func TestMalformedProjectionPreservesOutput(t *testing.T) {
	out := []float32{17}
	if err := ProjectQ4_0x8Q8_0Stage(nil, nil, int(^uint(0)>>1), 4, 1, out); err == nil {
		t.Fatal("wrapped rows")
	}
	if err := ProjectQ4_0x8Q8_0StageRows(nil, nil, 8, int(^uint(0)>>1), 9, 4, 1, out); err == nil {
		t.Fatal("wrapped groups")
	}
	if out[0] != 17 {
		t.Fatal("mutated output")
	}
}
