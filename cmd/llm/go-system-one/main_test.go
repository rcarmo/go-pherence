package main

import (
	"github.com/rcarmo/go-pherence/model"
	"testing"
)

func TestPackedDefaultAndOverrides(t *testing.T) {
	for _, tc := range []struct {
		args []string
		want int
		bad  bool
	}{
		{want: model.Gemma4PackedRows},
		{args: []string{"-backend", "simd"}, want: 0},
		{args: []string{"-packed-token-rows", "0"}, want: 0},
		{args: []string{"-packed-token-rows", "128"}, want: 128},
		{args: []string{"-packed-token-rows", "-1"}, want: model.Gemma4PackedRows},
		{args: []string{"-backend", "simd", "-packed-token-rows", "128"}, bad: true},
		{args: []string{"-packed-token-rows", "513"}, bad: true},
		{args: []string{"-packed-token-rows", "-2"}, bad: true},
	} {
		args := append([]string{"-model", "fixture", "-tokenizer-dir", "fixture"}, tc.args...)
		got, err := parseOptions(args)
		if (err != nil) != tc.bad {
			t.Fatalf("%v error=%v", tc.args, err)
		}
		if !tc.bad && got.packedRows != tc.want {
			t.Fatalf("%v rows=%d want=%d", tc.args, got.packedRows, tc.want)
		}
	}
}
