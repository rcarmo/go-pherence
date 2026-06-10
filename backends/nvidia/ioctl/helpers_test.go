package nv

import (
	"github.com/rcarmo/go-pherence/internal/checked"
	"testing"
)

func TestCheckedMulInt(t *testing.T) {
	if got, ok := checked.MulInt(6, 7); !ok || got != 42 {
		t.Fatalf("checked.MulInt(6,7)=%d,%v want 42,true", got, ok)
	}
	if _, ok := checked.MulInt(-1, 7); ok {
		t.Fatal("checkedMulInt accepted negative lhs")
	}
	if _, ok := checked.MulInt(7, -1); ok {
		t.Fatal("checkedMulInt accepted negative rhs")
	}
	maxInt := int(^uint(0) >> 1)
	if _, ok := checked.MulInt(maxInt/2+1, 3); ok {
		t.Fatal("checkedMulInt accepted overflow")
	}
}
