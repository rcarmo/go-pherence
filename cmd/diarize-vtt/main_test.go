package main

import "testing"

func TestParseVTTMillisAndFilterCompletedJobs(t *testing.T) {
	ms, ok := parseVTTMillis("01:02:03.456")
	if !ok || ms != 3723456 {
		t.Fatalf("parse=%d,%v", ms, ok)
	}
	jobs := []job{
		{idx: 0, cueStart: 0, cueEnd: 16000},
		{idx: 1, cueStart: 16000, cueEnd: 32000},
	}
	remaining := filterCompletedJobs(jobs, map[cueKey]bool{{startMS: 0, endMS: 1000}: true})
	if len(remaining) != 1 || remaining[0].idx != 1 {
		t.Fatalf("remaining=%+v", remaining)
	}
}

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
