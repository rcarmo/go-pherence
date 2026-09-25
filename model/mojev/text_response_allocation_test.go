package mojev

import (
	"reflect"
	"strings"
	"testing"
)

func TestPreparedTextDecisionSinglePackMatchesPublicAssembler(t *testing.T) {
	req, err := DecodeTextRequest(strings.NewReader(`{"model":"m","state":"ticket needs review","questions":{"priority":{"type":"choice","instructions":"Pick one.","criteria":{"z":"zebra","a":"ant","m":"moose"}},"blocked":{"type":"noul","criteria":{"false":"clear","true":"blocked"}},"score":{"type":"score","criteria":["low","medium","high"]}}}`))
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := prepareTextDecision(req, nil)
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	encode := func(text string) ([]int, error) {
		calls++
		return []int{len(text)%13 + 1, len(text)%7 + 20}, nil
	}
	packed, err := prepared.pack(128, 32, 0, encode)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 12 {
		t.Fatalf("pack encode calls=%d want 12", calls)
	}
	logits := [][]float64{{0.5, 0.1, 2.5}, {-1, 1}, {2, -3, 0.25}}
	got, err := prepared.assemble(logits, packed.PackedMask[0])
	if err != nil {
		t.Fatal(err)
	}
	if calls != 12 {
		t.Fatalf("assemble re-packed: encode calls=%d", calls)
	}
	publicCalls := 0
	want, err := AssembleTextDecision(req, logits, func(text string) ([]int, error) {
		publicCalls++
		return []int{len(text)%13 + 1, len(text)%7 + 20}, nil
	}, 0, 128, 32)
	if err != nil {
		t.Fatal(err)
	}
	if publicCalls != 12 {
		t.Fatalf("public encode calls=%d want 12", publicCalls)
	}
	if got.Usage.InputTokens != 24 {
		t.Fatalf("usage input tokens=%d want 24", got.Usage.InputTokens)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("prepared decision mismatch:\n got=%#v\nwant=%#v", got, want)
	}
}

func TestAssembleTextDecisionValidatesBeforeEncode(t *testing.T) {
	req, err := DecodeTextRequest(strings.NewReader(`{"model":"m","state":"s","questions":{"priority":{"type":"choice","criteria":{"x":"xray","a":"alpha"}}}}`))
	if err != nil {
		t.Fatal(err)
	}
	bad := req
	bad.Fields = append([]TextRequestField(nil), req.Fields...)
	bad.Fields[0].SortedIndices = append([]int(nil), req.Fields[0].SortedIndices...)
	bad.Fields[0].SortedOptions = append([]string(nil), req.Fields[0].SortedOptions...)
	bad.Fields[0].SortedIndices[0], bad.Fields[0].SortedIndices[1] = bad.Fields[0].SortedIndices[1], bad.Fields[0].SortedIndices[0]
	bad.Fields[0].SortedOptions[0], bad.Fields[0].SortedOptions[1] = bad.Fields[0].SortedOptions[1], bad.Fields[0].SortedOptions[0]
	calls := 0
	got, err := AssembleTextDecision(bad, [][]float64{{1, 2}}, func(string) ([]int, error) {
		calls++
		return []int{1}, nil
	}, 0, 128, 32)
	if err == nil || got != nil {
		t.Fatalf("accepted malformed decision: %+v", got)
	}
	if calls != 0 {
		t.Fatalf("encode called before validation: %d", calls)
	}
}

func TestPreparedDecisionRejectsInvalidLabelsBeforeInference(t *testing.T) {
	for _, kind := range []string{"unknown", "noul", "score"} {
		req, err := DecodeTextRequest(strings.NewReader(`{"model":"m","state":"s","questions":{"q":{"type":"choice","criteria":{"a":"alpha","b":"beta"}}}}`))
		if err != nil {
			t.Fatal(err)
		}
		req.Fields[0].Kind = kind
		if p, e := prepareTextDecision(req, nil); e == nil || p != nil {
			t.Fatal("bad kind/labels accepted", kind)
		}
	}
	req, err := DecodeTextRequest(strings.NewReader(`{"model":"m","state":"s","questions":{"q":{"type":"choice","criteria":{"a":"alpha","b":"beta"}}}}`))
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	if out, e := AssembleTextDecision(req, nil, func(string) ([]int, error) { calls++; return []int{1}, nil }, 0, 32, 32); e == nil || out != nil || calls != 0 {
		t.Fatal("nil logits reached encoder")
	}
}
