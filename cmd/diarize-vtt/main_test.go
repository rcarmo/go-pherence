package main

import "testing"

func TestDynamicMaxTokens(t *testing.T) {
	if got := dynamicMaxTokens(40, 0.5, 4); got != 12 {
		t.Fatalf("short cue budget=%d want 12", got)
	}
	if got := dynamicMaxTokens(40, 5, 4); got != 28 {
		t.Fatalf("5s cue budget=%d want 28", got)
	}
	if got := dynamicMaxTokens(40, 20, 4); got != 40 {
		t.Fatalf("capped budget=%d want 40", got)
	}
	if got := dynamicMaxTokens(40, 5, 0); got != 40 {
		t.Fatalf("disabled budget=%d want 40", got)
	}
}
