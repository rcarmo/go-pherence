package model

import (
	"context"
	"fmt"
	"testing"
	"time"
)

func TestPrefixAdmissionBudgetsAndRecovery(t *testing.T) {
	b := PrefixBranch{ID: "q", ValidSuffixTokens: 2, Suffix: []int{1, 2}, CandidateIDs: []string{"a", "b"}, CandidateTokens: []int{3, 4}}
	if _, e := validatePrefixBranches(10, 512, 100, []PrefixBranch{b}); e != nil {
		t.Fatal(e)
	}
	for _, bad := range []PrefixBranch{{ID: "q", Suffix: nil, CandidateIDs: b.CandidateIDs, CandidateTokens: b.CandidateTokens}, {ID: "q", ValidSuffixTokens: 1, Suffix: []int{-1}, CandidateIDs: b.CandidateIDs, CandidateTokens: b.CandidateTokens}, {ID: "q", ValidSuffixTokens: 503, Suffix: make([]int, 503), CandidateIDs: b.CandidateIDs, CandidateTokens: b.CandidateTokens}, {ID: "q", ValidSuffixTokens: 2, Suffix: b.Suffix, CandidateIDs: []string{"a", "a"}, CandidateTokens: b.CandidateTokens}, {ID: "padded", ValidSuffixTokens: 1, Suffix: []int{1, 0}, CandidateIDs: b.CandidateIDs, CandidateTokens: b.CandidateTokens}} {
		if _, e := validatePrefixBranches(10, 512, 100, []PrefixBranch{bad}); e == nil {
			t.Fatal("bad branch accepted")
		}
	}
	if _, e := validatePrefixBranches(10, 512, 100, []PrefixBranch{b, b}); e == nil {
		t.Fatal("duplicate stable ID")
	}
	e := &FrozenGPUEncoder{}
	release, err := e.reservePrefixWork(prefixWork{32, 2048, 4096, 512})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = e.reservePrefixWork(prefixWork{1, 1, 1, 2}); err == nil {
		t.Fatal("unbounded queue")
	}
	release()
	release()
	if e.prefixQueued != (prefixWork{}) {
		t.Fatal("admission leak")
	}
	release, err = e.reservePrefixWork(prefixWork{1, 1, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	release()
	p := &FrozenPrefix{owner: &FrozenGPUEncoder{maxTokens: 512, cfg: LlamaConfig{VocabSize: 100}}, ids: []int{1}}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if r, _, err := p.ScoreSuffixes(ctx, []PrefixBranch{b}, true); err == nil || r != nil {
		t.Fatal("pre-cancel not rejected")
	}
}

func TestPrefixCancelledQueueDoesNotRetainWork(t *testing.T) {
	e := &FrozenGPUEncoder{maxTokens: 512, cfg: LlamaConfig{VocabSize: 100}}
	e.prefixGateOnce.Do(func() { e.prefixGate = make(chan struct{}, 1) })
	e.prefixGate <- struct{}{}
	p := &FrozenPrefix{owner: e, ids: []int{1}}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
	defer cancel()
	b := PrefixBranch{ID: "queued", ValidSuffixTokens: 1, Suffix: []int{1}, CandidateIDs: []string{"a", "b"}, CandidateTokens: []int{2, 3}}
	if r, _, err := p.ScoreSuffixes(ctx, []PrefixBranch{b}, true); err == nil || r != nil {
		t.Fatal("queued cancellation not propagated")
	}
	<-e.prefixGate
	if e.prefixQueued != (prefixWork{}) {
		t.Fatal("queued cancellation leaked admission", e.prefixQueued)
	}
}

func TestPrefixAggregateAdmissionDimensions(t *testing.T) {
	makeBranches := func(n, length, candidates int) []PrefixBranch {
		out := make([]PrefixBranch, n)
		for i := range out {
			ids := make([]string, candidates)
			tokens := make([]int, candidates)
			for j := range ids {
				ids[j] = fmt.Sprintf("c-%d", j)
				tokens[j] = j
			}
			out[i] = PrefixBranch{ID: fmt.Sprintf("q-%d", i), ValidSuffixTokens: length, Suffix: make([]int, length), CandidateIDs: ids, CandidateTokens: tokens}
		}
		return out
	}
	for _, x := range []struct {
		prefix   int
		branches []PrefixBranch
	}{{1, makeBranches(8, 65, 2)}, {300, makeBranches(7, 1, 2)}, {1, makeBranches(5, 1, 32)}, {1, makeBranches(9, 1, 2)}} {
		if _, err := validatePrefixBranches(x.prefix, 512, 100, x.branches); err == nil {
			t.Fatal("aggregate dimension not enforced")
		}
	}
	for _, w := range []prefixWork{{33, 1, 1, 2}, {1, 2049, 1, 2}, {1, 1, 4097, 2}, {1, 1, 1, 513}} {
		e := &FrozenGPUEncoder{}
		if _, err := e.reservePrefixWork(w); err == nil {
			t.Fatal("queued dimension not enforced", w)
		}
	}
}
