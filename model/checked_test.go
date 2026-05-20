package model

import "testing"

func TestCheckedProduct(t *testing.T) {
	if got, ok := checkedProduct(3, 5); !ok || got != 15 {
		t.Fatalf("checkedProduct(3,5)=%d,%v want 15,true", got, ok)
	}
	if _, ok := checkedProduct(-1, 5); ok {
		t.Fatal("checkedProduct accepted negative lhs")
	}
	if _, ok := checkedProduct(5, -1); ok {
		t.Fatal("checkedProduct accepted negative rhs")
	}
	maxInt := int(^uint(0) >> 1)
	if _, ok := checkedProduct(maxInt/2+1, 3); ok {
		t.Fatal("checkedProduct accepted overflowing product")
	}
}
