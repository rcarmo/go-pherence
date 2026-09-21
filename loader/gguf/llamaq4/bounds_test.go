package llamaq4

import "testing"

func TestMalformedInputsNeverEnterC(t *testing.T) {
	if err := DotQ4_0x8Q8_0x4VNNI(make([]byte, 144), make([]byte, 136), 1, nil); err == nil {
		t.Fatal("nil output")
	}
	var out [32]float32
	if err := DotQ4_0x8Q8_0x4VNNI(nil, nil, int(^uint(0)>>1)/8+1, &out); err == nil {
		t.Fatal("wrapped buffers")
	}
	sentinel := []float32{17}
	if err := ProjectQ4_0x8Q8_0x4VNNI(nil, nil, int(^uint(0)>>1), 4, 1, sentinel); err == nil {
		t.Fatal("wrapped rows")
	}
	if err := ProjectQ4_0x8Q8_0x4RowsVNNI(nil, nil, 8, int(^uint(0)>>1), 9, 4, 1, sentinel); err == nil {
		t.Fatal("wrapped groups")
	}
	if sentinel[0] != 17 {
		t.Fatal("mutated output")
	}
}
