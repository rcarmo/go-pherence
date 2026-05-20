package nv

import "testing"

func TestCheckedMulInt(t *testing.T) {
	if got, ok := checkedMulInt(6, 7); !ok || got != 42 {
		t.Fatalf("checkedMulInt(6,7)=%d,%v want 42,true", got, ok)
	}
	if _, ok := checkedMulInt(-1, 7); ok {
		t.Fatal("checkedMulInt accepted negative lhs")
	}
	if _, ok := checkedMulInt(7, -1); ok {
		t.Fatal("checkedMulInt accepted negative rhs")
	}
	maxInt := int(^uint(0) >> 1)
	if _, ok := checkedMulInt(maxInt/2+1, 3); ok {
		t.Fatal("checkedMulInt accepted overflow")
	}
}
