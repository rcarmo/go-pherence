package main

import "testing"

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
