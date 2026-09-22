package pockettts

import "testing"

func TestPrepareText(t *testing.T) {
	for _, tc := range []struct {
		in, out string
		after   int
	}{{"hello world", "Hello world.", 3}, {"Already done!", "Already done!", 3}, {"one two three four five", "One two three four five.", 1}, {"quoted\ntext,", "Quoted text.", 3}} {
		got, after, err := PrepareText(tc.in)
		if err != nil || got != tc.out || after != tc.after {
			t.Fatalf("PrepareText(%q)=%q,%d,%v", tc.in, got, after, err)
		}
	}
	if _, _, err := PrepareText("  "); err == nil {
		t.Fatal("accepted empty text")
	}
}
