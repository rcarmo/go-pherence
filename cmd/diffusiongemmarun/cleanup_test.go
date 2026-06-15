package main

import "testing"

func TestRunFatalCleanupsLIFOAndClears(t *testing.T) {
	fatalCleanups = nil
	var got []int
	registerFatalCleanup(func() { got = append(got, 1) })
	registerFatalCleanup(func() { got = append(got, 2) })
	registerFatalCleanup(func() { panic("cleanup panic should be recovered") })
	registerFatalCleanup(func() { got = append(got, 3) })
	runFatalCleanups()
	want := []int{3, 2, 1}
	if len(got) != len(want) {
		t.Fatalf("cleanup order len=%d want %d got=%v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("cleanup order got=%v want=%v", got, want)
		}
	}
	if len(fatalCleanups) != 0 {
		t.Fatalf("fatalCleanups not cleared: %d", len(fatalCleanups))
	}
	runFatalCleanups()
	if len(got) != len(want) {
		t.Fatalf("cleanup reran after clear: got=%v", got)
	}
}
