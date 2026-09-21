package main

import (
	"bytes"
	"reflect"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/model/jevlike"
)

func TestOnlineScenarioCallsAndExactLogits(t *testing.T) {
	contract := frozenFeatureContract("synthetic#sha256="+strings.Repeat("a", 64), strings.Repeat("b", 64), "f32")
	contract.Width = 4
	calls := 0
	live := &jevlike.FeatureEncoder{Contract: contract, Tokenize: func(s string) ([]int, error) { return []int{len(s), 1}, nil }, EncodeTokens: func(ids []int) ([][]float32, error) {
		calls++
		return [][]float32{{float32(ids[0]), 2, 3, 4}, {1, 3, 2, 4}}, nil
	}}
	ex := jevlike.ChoiceExample{Context: "The key is in the blue drawer.", Options: []string{"The key is in the blue drawer.", "The key is in the red drawer."}, Label: 0}
	context, e := live.Encode(ex.Context, 512)
	if e != nil {
		t.Fatal(e)
	}
	options := map[string][]float32{}
	for _, text := range ex.Options {
		options[text], e = live.EncodeOption(text, 128)
		if e != nil {
			t.Fatal(e)
		}
	}
	head, _ := jevlike.NewAttentionHead(4, 2)
	base := &jevlike.FrozenScorer{Config: jevlike.Config{Width: 4, Rank: 2, ContextTokens: 512, OptionTokens: 128}, Reference: live.FeatureReference(), Head: *head, Encoder: live}
	jevlike.InitializeFrozenScorer(base, 7)
	ck, e := jevlike.FrozenCheckpoint(base)
	if e != nil {
		t.Fatal(e)
	}
	var reference [][]float32
	for mode := 0; mode < 3; mode++ {
		source := &onlineFeatureEncoder{live: live, context: context, options: options, mode: mode}
		m, e := ck.Frozen(source)
		if e != nil {
			t.Fatal(e)
		}
		calls = 0
		got, e := m.Forward([]jevlike.ChoiceExample{ex}, false)
		if e != nil {
			t.Fatal(e)
		}
		want := []int{3, 1, 0}[mode]
		if calls != want {
			t.Fatal("wrong encoder calls", mode, calls)
		}
		if mode == 0 {
			reference = got
		} else if !reflect.DeepEqual(reference, got) {
			t.Fatal("reuse changed logits")
		}
	}
}
func TestHeadCheckCLIHelpAndMissingPaths(t *testing.T) {
	for _, name := range []string{"head-permutation", "head-online-bench", "feature-score"} {
		var out, stderr bytes.Buffer
		if e := run([]string{name, "-h"}, &out, &stderr); e != nil {
			t.Fatal(e)
		}
		if e := run([]string{name}, &out, &stderr); e == nil {
			t.Fatal("missing paths admitted")
		}
	}
}
