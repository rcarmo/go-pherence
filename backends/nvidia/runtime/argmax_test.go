package nvidia

import "testing"

func TestArgmaxF32Malformed(t *testing.T) {
	if _, _, err := ArgmaxF32(nil, 1); err == nil {
		t.Fatalf("expected nil-buffer error")
	}
	buf := &Buffer{Size: 4}
	if _, _, err := ArgmaxF32(buf, 0); err == nil {
		t.Fatalf("expected invalid length error")
	}
	if _, _, err := ArgmaxF32(buf, 2); err == nil {
		t.Fatalf("expected length exceeds buffer error")
	}
}
