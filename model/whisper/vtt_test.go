package whisper

import (
	"bytes"
	"strings"
	"testing"
)

func TestWriteVTT(t *testing.T) {
	segments := []Segment{
		{Start: 0.0, End: 2.5, Text: "Hello world"},
		{Start: 3.0, End: 5.5, Text: "How are you"},
		{Start: 60.5, End: 63.123, Text: "One minute later"},
	}

	var buf bytes.Buffer
	WriteVTTTo(&buf, segments)
	output := buf.String()

	if !strings.HasPrefix(output, "WEBVTT\n\n") {
		t.Fatalf("missing WEBVTT header: %q", output[:20])
	}
	if !strings.Contains(output, "00:00:00.000 --> 00:00:02.500") {
		t.Fatalf("wrong timestamp format in:\n%s", output)
	}
	if !strings.Contains(output, "00:01:00.500 --> 00:01:03.12") {
		t.Fatalf("wrong minute timestamp in:\n%s", output)
	}
	if !strings.Contains(output, "Hello world") {
		t.Fatal("missing text")
	}
	t.Logf("VTT output:\n%s", output)
}

func TestWriteDiarizedVTT(t *testing.T) {
	segments := []DiarizedSegment{
		{Start: 0.0, End: 2.0, Speaker: 0, Text: "Hello"},
		{Start: 2.5, End: 4.0, Speaker: 1, Text: "Hi there"},
		{Start: 4.5, End: 6.0, Speaker: 0, Text: "How are you"},
	}

	var buf bytes.Buffer
	WriteDiarizedVTTTo(&buf, segments)
	output := buf.String()

	if !strings.HasPrefix(output, "WEBVTT\n\n") {
		t.Fatalf("missing WEBVTT header")
	}
	if !strings.Contains(output, "<v Speaker 1>Hello") {
		t.Fatalf("missing speaker 1 tag in:\n%s", output)
	}
	if !strings.Contains(output, "<v Speaker 2>Hi there") {
		t.Fatalf("missing speaker 2 tag in:\n%s", output)
	}
	t.Logf("Diarized VTT output:\n%s", output)
}

func TestFormatVTTTime(t *testing.T) {
	cases := []struct {
		input float64
		want  string
	}{
		{0, "00:00:00.000"},
		{1.5, "00:00:01.500"},
		{61.234, "00:01:01.234"},
		{3661.5, "01:01:01.500"},
	}
	for _, tc := range cases {
		got := formatVTTTime(tc.input)
		if got != tc.want {
			t.Fatalf("formatVTTTime(%f) = %q want %q", tc.input, got, tc.want)
		}
	}
}
