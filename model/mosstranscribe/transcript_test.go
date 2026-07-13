package mosstranscribe

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestTranscriptParserStreamingAndBracketText(t *testing.T) {
	parser := NewTranscriptParser()
	chunks := []string{"noise[0.00][S", "01] Hello [wor", "ld] [1.25] \n[1.25][S02]Next", "[2]"}
	var got []TranscriptSegment
	for _, chunk := range chunks {
		got = append(got, parser.Feed(chunk)...)
	}
	got = append(got, parser.Close()...)
	want := []TranscriptSegment{
		{Start: 0, End: 1.25, Speaker: "S01", Text: "Hello [world]"},
		{Start: 1.25, End: 2, Speaker: "S02", Text: "Next"},
	}
	if len(got) != len(want) {
		t.Fatalf("segments=%+v want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("segment[%d]=%+v want %+v", i, got[i], want[i])
		}
	}
}

func TestTranscriptParserRejectsMalformedAndSkipsEmpty(t *testing.T) {
	for _, text := range []string{"[bad][S01]x[1]", "[2][speaker]x[3]", "[3][S03]x[2]", "[4][S04]   [5]"} {
		if got := ParseTranscript(text); len(got) != 0 {
			t.Fatalf("ParseTranscript(%q)=%+v", text, got)
		}
	}
	got := ParseTranscript("noise [6][S06]ok[7]")
	if len(got) != 1 || got[0].Speaker != "S06" || got[0].Text != "ok" {
		t.Fatalf("segments=%+v", got)
	}
	parser := NewTranscriptParser()
	if got := parser.Feed("[0][S01]unfinished"); len(got) != 0 {
		t.Fatalf("emitted incomplete segment %+v", got)
	}
	if got := parser.Close(); len(got) != 0 {
		t.Fatalf("closed incomplete segment %+v", got)
	}
}

func TestTranscriptExportSurfaces(t *testing.T) {
	segments := []TranscriptSegment{{Start: 1.2345, End: 65.789, Speaker: "S01", Text: "Hello\nworld"}}
	jsonData, err := TranscriptJSON(segments)
	if err != nil {
		t.Fatal(err)
	}
	var decoded []TranscriptSegment
	if err := json.Unmarshal(jsonData, &decoded); err != nil || len(decoded) != 1 || decoded[0] != segments[0] {
		t.Fatalf("JSON round trip decoded=%+v err=%v", decoded, err)
	}
	srt := TranscriptSRT(segments)
	if !strings.Contains(srt, "00:00:01,235 --> 00:01:05,789") || !strings.Contains(srt, "[S01] Hello\nworld") {
		t.Fatalf("SRT:\n%s", srt)
	}
	ass := TranscriptASS(segments)
	if !strings.Contains(ass, "Dialogue: 0,00:00:01.23,00:01:05.79,Default,S01") || !strings.Contains(ass, "Hello\\Nworld") {
		t.Fatalf("ASS:\n%s", ass)
	}
}
