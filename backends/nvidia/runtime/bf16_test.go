package nvidia

import "testing"

func TestCheckedBF16MatrixBytes(t *testing.T) {
	got, ok := checkedBF16MatrixBytes(3, 5)
	if !ok || got != 30 {
		t.Fatalf("checkedBF16MatrixBytes(3,5)=%d,%v want 30,true", got, ok)
	}
	if _, ok := checkedBF16MatrixBytes(0, 5); ok {
		t.Fatal("checkedBF16MatrixBytes accepted zero rows")
	}
	maxInt := int(^uint(0) >> 1)
	if _, ok := checkedBF16MatrixBytes(maxInt/2+1, 2); ok {
		t.Fatal("checkedBF16MatrixBytes accepted overflowing byte count")
	}
	if _, ok := checkedBF16MatrixBytes(maxInt, 2); ok {
		t.Fatal("checkedBF16MatrixBytes accepted overflowing element count")
	}
}

func TestBF16LMHeadWithBufferRejectsShortWeightBuffer(t *testing.T) {
	logits := make([]float32, 4)
	x := make([]float32, 8)
	short := &Buffer{Size: 4}
	if err := BF16LMHeadWithBuffer(logits, short, x, 4, 8); err != nil {
		t.Fatalf("BF16LMHeadWithBuffer short buffer returned error %v, want nil fallback", err)
	}
}
