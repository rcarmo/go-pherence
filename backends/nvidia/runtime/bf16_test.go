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

func TestValidBF16BufferRejectsMalformedInputs(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	if validBF16Buffer(nil, 1) {
		t.Fatal("validBF16Buffer accepted nil buffer")
	}
	if validBF16Buffer(&Buffer{Ptr: 1, Size: 2}, 0) {
		t.Fatal("validBF16Buffer accepted zero n")
	}
	if validBF16Buffer(&Buffer{Ptr: 1, Size: 2}, maxInt/2+1) {
		t.Fatal("validBF16Buffer accepted overflowing n")
	}
	if validBF16Buffer(&Buffer{Ptr: 1, Size: 2}, 2) {
		t.Fatal("validBF16Buffer accepted short buffer")
	}
	if !validBF16Buffer(&Buffer{Ptr: 1, Size: 4}, 2) {
		t.Fatal("validBF16Buffer rejected valid buffer")
	}
}

func TestBF16DeviceWrappersRejectMalformedInputs(t *testing.T) {
	oldRMS, oldNoScale, oldAdd, oldSiLU, oldGELU := fnBF16RMSNorm, fnBF16RMSNormNoScale, fnBF16VecAdd, fnBF16SiLUMul, fnBF16GELUTanhMul
	defer func() {
		fnBF16RMSNorm, fnBF16RMSNormNoScale, fnBF16VecAdd, fnBF16SiLUMul, fnBF16GELUTanhMul = oldRMS, oldNoScale, oldAdd, oldSiLU, oldGELU
	}()
	fnBF16RMSNorm, fnBF16RMSNormNoScale, fnBF16VecAdd, fnBF16SiLUMul, fnBF16GELUTanhMul = 1, 1, 1, 1, 1
	short := &Buffer{Ptr: 1, Size: 2}
	valid := &Buffer{Ptr: 1, Size: 4}
	if DevBF16RMSNorm(valid, short, 2, 1e-6) {
		t.Fatal("DevBF16RMSNorm accepted short weight")
	}
	if DevBF16RMSNormNoScale(short, 2, 1e-6) {
		t.Fatal("DevBF16RMSNormNoScale accepted short input")
	}
	if DevBF16VecAdd(valid, valid, short, 2) {
		t.Fatal("DevBF16VecAdd accepted short input")
	}
	if DevBF16SiLUMul(valid, valid, short, 2) {
		t.Fatal("DevBF16SiLUMul accepted short input")
	}
	if DevBF16GELUTanhMul(valid, short, 2) {
		t.Fatal("DevBF16GELUTanhMul accepted short input")
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
