package main

import "testing"

func TestGeneratedSuffixFromFullOutput(t *testing.T) {
	if got := generatedSuffixFromFullOutput(2, 3, []int{10, 11, 12, 13, 14}); !sameIntsForLLMGenTest(got, []int{12, 13, 14}) {
		t.Fatalf("unwrapped suffix=%v", got)
	}
	if got := generatedSuffixFromFullOutput(10, 1, []int{2, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109}); !sameIntsForLLMGenTest(got, []int{109}) {
		t.Fatalf("templated suffix=%v", got)
	}
	if got := generatedSuffixFromFullOutput(3, 0, []int{1, 2, 3}); len(got) != 0 {
		t.Fatalf("zero-token suffix=%v", got)
	}
}

func sameIntsForLLMGenTest(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestMTPEffectivePromptTokenCount(t *testing.T) {
	if got := mtpEffectivePromptTokenCount(1, 12, true, 1, 1); got != 10 {
		t.Fatalf("MTP prompt tokens=%d want templated prompt 10", got)
	}
	if got := mtpEffectivePromptTokenCount(3, 5, false, 99, 99); got != 3 {
		t.Fatalf("regular prompt tokens=%d want input ids", got)
	}
	if got := mtpEffectivePromptTokenCount(3, 2, true, 9, 9); got != 3 {
		t.Fatalf("invalid MTP accounting prompt tokens=%d want fallback input ids", got)
	}
}

func TestFormatMTPFinalStateCoverage(t *testing.T) {
	if got := formatMTPFinalStateCoverage(4, 4); got != "4/4 tokens" {
		t.Fatalf("coverage=%q", got)
	}
	if got := formatMTPFinalStateCoverage(3, 4); got != "3/4 tokens (greedy tail not covered)" {
		t.Fatalf("tail coverage=%q", got)
	}
}

func TestValidateMTPCLIFlags(t *testing.T) {
	cases := []struct {
		name        string
		tokens      int
		mtpSmoke    bool
		mtpGenerate bool
		drafter     string
		seq         int
		wantErr     bool
	}{
		{name: "regular", tokens: 1, seq: 1},
		{name: "smoke with drafter", tokens: 0, mtpSmoke: true, drafter: "assistant", seq: 1},
		{name: "generate with drafter", tokens: 4, mtpGenerate: true, drafter: "assistant", seq: 1},
		{name: "negative tokens", tokens: -1, seq: 1, wantErr: true},
		{name: "smoke missing drafter", tokens: 1, mtpSmoke: true, seq: 1, wantErr: true},
		{name: "generate missing drafter", tokens: 1, mtpGenerate: true, seq: 1, wantErr: true},
		{name: "bad seq", tokens: 1, seq: 0, wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateMTPCLIFlags(tc.tokens, tc.mtpSmoke, tc.mtpGenerate, tc.drafter, tc.seq)
			if (err != nil) != tc.wantErr {
				t.Fatalf("validateMTPCLIFlags err=%v wantErr=%v", err, tc.wantErr)
			}
		})
	}
}

func TestMTPKVExtentValidation(t *testing.T) {
	for _, dims := range [][2]int{{0, 1}, {1, 0}, {-1, 4}, {int(^uint(0) >> 1), 4}, {int(^uint(0)>>1) / 2, 1}} {
		if _, err := mtpKVElements(dims[0], dims[1]); err == nil {
			t.Fatal(dims)
		}
	}
	if n, err := mtpKVElements(3, 4); err != nil || n != 12 {
		t.Fatal(n, err)
	}
}

func TestPreparedWrapperCannotBecomeGeneratedSuffix(t *testing.T) {
	// Ten prepared tokens and one result with a generous budget: the old raw
	// count plus budget heuristic incorrectly labelled wrapper tokens as output.
	out := []int{2, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109}
	if got := generatedSuffixFromFullOutput(10, 50, out); !sameIntsForLLMGenTest(got, []int{109}) {
		t.Fatal(got)
	}
	if got := generatedSuffixFromFullOutput(10, 50, out[:10]); len(got) != 0 {
		t.Fatal("wrapper emitted", got)
	}
	if got := generatedSuffixFromFullOutput(12, 50, out); len(got) != 0 {
		t.Fatal("short failed output emitted", got)
	}
}
