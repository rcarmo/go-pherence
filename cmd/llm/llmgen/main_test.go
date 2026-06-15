package main

import "testing"

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
