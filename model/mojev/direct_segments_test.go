package mojev

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestDirectSegmentsEncodeMatchesPackedExtraction(t *testing.T) {
	fields := []TextField{
		{Name: "route_name", Description: "Pick the scenic detour.", Options: []string{"amber", "crimson", "violet"}},
		{Name: "risk_level", Description: "Rate the operational risk.", Options: []string{"low", "high"}},
	}
	row := TextRow{State: "state-segment", Menus: [][]string{nil, nil}}
	prompt0, err := renderChoicePrompt(fields[0])
	if err != nil {
		t.Fatal(err)
	}
	prompt1, err := renderChoicePrompt(fields[1])
	if err != nil {
		t.Fatal(err)
	}
	largeID := int(^uint(0)>>1) - 7
	encode := directSegmentsMapEncoder(map[string][]int{
		row.State: {0, 11, 12, 13, 14},
		prompt0:   {101, 102, 103, 104},
		prompt1:   {201, 202, 203},
		"amber":   {301, 302, 303},
		"crimson": {largeID, 304, 305, 306},
		"violet":  {307},
		"low":     {401, 402, 403, 404},
		"high":    {405},
	})

	encoded, err := encodeTextRows([]TextRow{row}, fields, 3, 2, encode)
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded) != 1 {
		t.Fatalf("rows=%d want 1", len(encoded))
	}
	got := encoded[0]
	want := EncodedRow{
		State:     []int{0, 11, 12},
		Questions: [][]int{{101, 102}, {201, 202}},
		Candidates: [][][]int{
			{{301, 302, 303}, {largeID, 304, 305, 306}, {307}},
			{{401, 402, 403, 404}, {405}},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("direct encode mismatch:\n got=%#v\nwant=%#v", got, want)
	}
	if len(got.Candidates[0][1]) != 4 || got.Candidates[0][1][0] != largeID {
		t.Fatalf("candidate truncation changed: %v", got.Candidates[0][1])
	}
	if len(got.Candidates[1][0]) != 4 {
		t.Fatalf("candidate length truncated: %v", got.Candidates[1][0])
	}

	packed, err := PackTextRows([]TextRow{row}, fields, 3, 2, 99, encode)
	if err != nil {
		t.Fatal(err)
	}
	old := directSegmentsExtractEncodedRow(t, packed, 0)
	if !reflect.DeepEqual(old, got) {
		t.Fatalf("packed extraction mismatch:\n got=%#v\nwant=%#v", old, got)
	}
	if tokens, packedTokens := encodedRowTokenCount(got), directSegmentsCountTrue(packed.PackedMask[0]); tokens != packedTokens {
		t.Fatalf("token count=%d want %d", tokens, packedTokens)
	}
}

func TestDirectSegmentsReferenceLimitAndIDValidation(t *testing.T) {
	fields := []TextField{
		{Name: "left", Options: []string{"aa", "bb"}},
		{Name: "right", Options: []string{"cc", "dd"}},
	}
	row := TextRow{State: "limit-state", Menus: [][]string{nil, nil}}
	prompt0, err := renderChoicePrompt(fields[0])
	if err != nil {
		t.Fatal(err)
	}
	prompt1, err := renderChoicePrompt(fields[1])
	if err != nil {
		t.Fatal(err)
	}

	t.Run("total4096accept", func(t *testing.T) {
		encode := directSegmentsMapEncoder(map[string][]int{
			row.State: directSegmentsFilled(1000, 4090),
			prompt0:   {1},
			prompt1:   {2},
			"aa":      {3},
			"bb":      {4},
			"cc":      {5},
			"dd":      {6},
		})
		encoded, err := encodeTextRows([]TextRow{row}, fields, 4096, 4096, encode)
		if err != nil {
			t.Fatal(err)
		}
		if len(encoded) != 1 || encodedRowTokenCount(encoded[0]) != 4096 {
			t.Fatalf("encoded rows=%d tokens=%d", len(encoded), encodedRowTokenCount(encoded[0]))
		}
		packed, err := PackTextRows([]TextRow{row}, fields, 4096, 4096, 0, encode)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(directSegmentsExtractEncodedRow(t, packed, 0), encoded[0]) {
			t.Fatal("4096-token direct and packed rows differ")
		}
	})

	t.Run("total4097reject", func(t *testing.T) {
		encode := directSegmentsMapEncoder(map[string][]int{
			row.State: directSegmentsFilled(1000, 4091),
			prompt0:   {1},
			prompt1:   {2},
			"aa":      {3},
			"bb":      {4},
			"cc":      {5},
			"dd":      {6},
		})
		if out, err := encodeTextRows([]TextRow{row}, fields, 4096, 4096, encode); err == nil || out != nil || !strings.Contains(err.Error(), "reference limit") {
			t.Fatalf("direct overflow got %v, %v", out, err)
		}
		if out, err := PackTextRows([]TextRow{row}, fields, 4096, 4096, 0, encode); err == nil || out != nil || !strings.Contains(err.Error(), "reference limit") {
			t.Fatalf("packed overflow got %v, %v", out, err)
		}
	})

	t.Run("negative_id_reject", func(t *testing.T) {
		encode := directSegmentsMapEncoder(map[string][]int{
			row.State: {1, 2, 3},
			prompt0:   {-1},
			prompt1:   {2},
			"aa":      {3},
			"bb":      {4},
			"cc":      {5},
			"dd":      {6},
		})
		if out, err := encodeTextRows([]TextRow{row}, fields, 4096, 4096, encode); err == nil || out != nil || !strings.Contains(err.Error(), "negative token ID") {
			t.Fatalf("direct negative ID got %v, %v", out, err)
		}
		if out, err := PackTextRows([]TextRow{row}, fields, 4096, 4096, 0, encode); err == nil || out != nil || !strings.Contains(err.Error(), "negative token ID") {
			t.Fatalf("packed negative ID got %v, %v", out, err)
		}
	})
}

func TestDirectSegmentsEncodingFailures(t *testing.T) {
	fields := []TextField{{Name: "f", Options: []string{"a", "b"}}}
	rows := []TextRow{{State: "state", Menus: [][]string{nil}}}
	boom := errors.New("encode failure")
	for name, encode := range map[string]TextEncoder{
		"empty":                       func(string) ([]int, error) { return nil, nil },
		"oversized_before_truncation": func(string) ([]int, error) { return make([]int, 4097), nil },
		"negative_after_truncation":   func(string) ([]int, error) { return []int{1, -1}, nil },
		"callback":                    func(string) ([]int, error) { return nil, boom },
	} {
		t.Run(name, func(t *testing.T) {
			out, err := encodeTextRows(rows, fields, 1, 1, encode)
			if err == nil || out != nil {
				t.Fatalf("accepted invalid segment: %v %v", out, err)
			}
			if name == "callback" && !errors.Is(err, boom) {
				t.Fatal("lost callback error", err)
			}
		})
	}
}

func TestDirectSegmentsScoreTextContextWithMatchesPackedResponse(t *testing.T) {
	req := decodeTextOrchestrationRequest(t, `{"model":"m","state":"alpha beta gamma delta epsilon zeta","questions":{"route":{"type":"choice","instructions":"Pick the scenic detour now.","criteria":{"z":"last","a":"first","m":"middle"}},"risk":{"type":"score","instructions":"Rate the operational risk carefully.","criteria":["low","high"]}}}`)
	tok := syntheticASCIITokenizer(req)
	const stateLimit = 10
	const questionLimit = 12
	prepared, err := prepareTextDecision(req, nil)
	if err != nil {
		t.Fatal(err)
	}
	encode := func(text string) ([]int, error) { return tok.Encode(text), nil }
	packed, err := prepared.pack(stateLimit, questionLimit, 0, encode)
	if err != nil {
		t.Fatal(err)
	}
	wantRow := directSegmentsExtractEncodedRow(t, packed, 0)
	wantLogits64 := [][]float64{{-4, 0, 4}, {6, -6}}
	want, err := AssembleSafeTextDecision(req, wantLogits64, tok, 0, stateLimit, questionLimit)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("packed_equivalence_owned_repeated", func(t *testing.T) {
		shared := [][]float32{{-4, 0, 4}, {6, -6}}
		var captured EncodedRow
		got1, err := scoreTextContextWith(context.Background(), req, tok, stateLimit, questionLimit, func(row EncodedRow) ([][]float32, error) {
			captured = directSegmentsCloneEncodedRow(row)
			return shared, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(captured, wantRow) {
			t.Fatalf("captured row mismatch:\n got=%#v\nwant=%#v", captured, wantRow)
		}
		if !reflect.DeepEqual(got1, want) {
			t.Fatalf("direct response mismatch:\n got=%#v\nwant=%#v", got1, want)
		}
		if got1.Usage.InputTokens != directSegmentsCountTrue(packed.PackedMask[0]) || got1.Usage.InputTokens != encodedRowTokenCount(captured) {
			t.Fatalf("usage mismatch: %+v packed=%d direct=%d", got1.Usage, directSegmentsCountTrue(packed.PackedMask[0]), encodedRowTokenCount(captured))
		}

		got2, err := scoreTextContextWith(context.Background(), req, tok, stateLimit, questionLimit, func(row EncodedRow) ([][]float32, error) {
			if !reflect.DeepEqual(directSegmentsCloneEncodedRow(row), wantRow) {
				t.Fatal("repeat row changed")
			}
			return shared, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got2, want) {
			t.Fatalf("repeat response mismatch:\n got=%#v\nwant=%#v", got2, want)
		}
		got1.Answers["route"]["choice"] = "mutated"
		if !reflect.DeepEqual(got2, want) {
			t.Fatalf("second response aliased first mutation:\n got=%#v\nwant=%#v", got2, want)
		}
	})

	t.Run("callback_error_no_partial_output", func(t *testing.T) {
		boom := errors.New("boom")
		var captured EncodedRow
		got, err := scoreTextContextWith(context.Background(), req, tok, stateLimit, questionLimit, func(row EncodedRow) ([][]float32, error) {
			captured = directSegmentsCloneEncodedRow(row)
			return nil, boom
		})
		if !errors.Is(err, boom) || got != nil {
			t.Fatalf("got=%#v err=%v", got, err)
		}
		if !reflect.DeepEqual(captured, wantRow) {
			t.Fatalf("captured row mismatch on error:\n got=%#v\nwant=%#v", captured, wantRow)
		}
	})

	t.Run("malformed_logits_no_partial_output", func(t *testing.T) {
		got, err := scoreTextContextWith(context.Background(), req, tok, stateLimit, questionLimit, func(row EncodedRow) ([][]float32, error) {
			if !reflect.DeepEqual(directSegmentsCloneEncodedRow(row), wantRow) {
				t.Fatal("malformed row changed")
			}
			return [][]float32{{1, 2, 3}, {4}}, nil
		})
		if err == nil || !strings.Contains(err.Error(), "logit row") || got != nil {
			t.Fatalf("got=%#v err=%v", got, err)
		}
	})
}

func BenchmarkDirectSegmentsEncodeVsPackedExtraction(b *testing.B) {
	fields := []TextField{
		{Name: "route_name", Description: "Pick the scenic detour and explain why.", Options: []string{"amber-lane", "crimson-highway", "violet-ferry", "silver-tunnel"}},
		{Name: "risk_level", Description: "Rate the operational risk for the selected route.", Options: []string{"minimal", "moderate", "significant"}},
	}
	row := TextRow{State: strings.Repeat("branch-state/", 48), Menus: [][]string{nil, nil}}
	prompt0, err := renderChoicePrompt(fields[0])
	if err != nil {
		b.Fatal(err)
	}
	prompt1, err := renderChoicePrompt(fields[1])
	if err != nil {
		b.Fatal(err)
	}
	base := map[string][]int{
		row.State:         directSegmentsFilled(1, 640),
		prompt0:           directSegmentsFilled(1001, 192),
		prompt1:           directSegmentsFilled(2001, 176),
		"amber-lane":      directSegmentsFilled(3001, 40),
		"crimson-highway": directSegmentsFilled(4001, 44),
		"violet-ferry":    directSegmentsFilled(5001, 36),
		"silver-tunnel":   directSegmentsFilled(6001, 48),
		"minimal":         directSegmentsFilled(7001, 24),
		"moderate":        directSegmentsFilled(8001, 28),
		"significant":     directSegmentsFilled(9001, 32),
	}
	encode := directSegmentsMapEncoder(base)

	b.Run("direct_encode_rows", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rows, err := encodeTextRows([]TextRow{row}, fields, 256, 128, encode)
			if err != nil {
				b.Fatal(err)
			}
			directSegmentsBenchRow = rows[0]
		}
	})

	b.Run("pack_and_extract", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			packed, err := PackTextRows([]TextRow{row}, fields, 256, 128, 0, encode)
			if err != nil {
				b.Fatal(err)
			}
			directSegmentsBenchRow = directSegmentsExtractEncodedRow(b, packed, 0)
		}
	})
}

var directSegmentsBenchRow EncodedRow

func directSegmentsMapEncoder(base map[string][]int) TextEncoder {
	return func(text string) ([]int, error) {
		ids, ok := base[text]
		if !ok {
			return nil, fmt.Errorf("unexpected text %q", text)
		}
		return append([]int(nil), ids...), nil
	}
}

func directSegmentsExtractEncodedRow(tb testing.TB, packed *PackedRows, row int) EncodedRow {
	tb.Helper()
	if packed == nil || row < 0 || row >= len(packed.IDs) {
		tb.Fatalf("bad packed row %d", row)
	}
	ids := packed.IDs[row]
	packedMask := packed.PackedMask[row]
	extract := func(mask []bool) []int {
		out := make([]int, 0, directSegmentsCountTrue(mask))
		for i, keep := range mask {
			if !keep {
				continue
			}
			if !packedMask[i] {
				tb.Fatalf("span outside packed mask at %d", i)
			}
			out = append(out, ids[i])
		}
		return out
	}
	out := EncodedRow{
		State:      extract(packed.State[row]),
		Questions:  make([][]int, len(packed.Questions[row])),
		Candidates: make([][][]int, len(packed.Candidates[row])),
	}
	for f := range packed.Questions[row] {
		out.Questions[f] = extract(packed.Questions[row][f])
		for n, present := range packed.OptionMask[row][f] {
			if !present {
				continue
			}
			out.Candidates[f] = append(out.Candidates[f], extract(packed.Candidates[row][f][n]))
		}
	}
	return out
}

func directSegmentsCloneEncodedRow(row EncodedRow) EncodedRow {
	out := EncodedRow{
		State:      append([]int(nil), row.State...),
		Questions:  make([][]int, len(row.Questions)),
		Candidates: make([][][]int, len(row.Candidates)),
	}
	for f, question := range row.Questions {
		out.Questions[f] = append([]int(nil), question...)
	}
	for f, group := range row.Candidates {
		out.Candidates[f] = make([][]int, len(group))
		for n, candidate := range group {
			out.Candidates[f][n] = append([]int(nil), candidate...)
		}
	}
	return out
}

func directSegmentsCountTrue(mask []bool) int {
	n := 0
	for _, present := range mask {
		if present {
			n++
		}
	}
	return n
}

func directSegmentsFilled(start, n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = start + i
	}
	return out
}
