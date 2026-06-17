package nvidia

import "testing"

func TestVecAddF32BufferRejectsBadInputs(t *testing.T) {
	valid := &Buffer{Ptr: 1, Size: 4}
	short := &Buffer{Ptr: 1, Size: 2}
	if err := VecAddF32Buffer(valid, valid, short, 1); err == nil {
		t.Fatal("accepted short output buffer")
	}
	if err := VecAddF32Buffer(nil, valid, valid, 1); err == nil {
		t.Fatal("accepted nil input buffer")
	}
	if err := VecAddF32Buffer(valid, valid, valid, 0); err != nil {
		t.Fatalf("zero-length add returned error: %v", err)
	}
}
