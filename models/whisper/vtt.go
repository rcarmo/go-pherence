package whisper

import (
	"fmt"
	"io"
	"os"
	"strings"
)

// DiarizedSegment combines transcription with speaker identity.
type DiarizedSegment struct {
	Start   float64
	End     float64
	Speaker int
	Text    string
}

// WriteVTT writes segments to a WebVTT file.
func WriteVTT(path string, segments []Segment) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return WriteVTTTo(f, segments)
}

// WriteVTTTo writes segments in WebVTT format to a writer.
func WriteVTTTo(w io.Writer, segments []Segment) error {
	fmt.Fprintln(w, "WEBVTT")
	fmt.Fprintln(w)

	for i, seg := range segments {
		fmt.Fprintf(w, "%d\n", i+1)
		fmt.Fprintf(w, "%s --> %s\n", formatVTTTime(seg.Start), formatVTTTime(seg.End))
		fmt.Fprintf(w, "%s\n\n", strings.TrimSpace(seg.Text))
	}
	return nil
}

// WriteDiarizedVTT writes diarized segments to a WebVTT file with speaker labels.
func WriteDiarizedVTT(path string, segments []DiarizedSegment) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return WriteDiarizedVTTTo(f, segments)
}

// WriteDiarizedVTTTo writes diarized segments in WebVTT format with speaker voice tags.
func WriteDiarizedVTTTo(w io.Writer, segments []DiarizedSegment) error {
	fmt.Fprintln(w, "WEBVTT")
	fmt.Fprintln(w)

	for i, seg := range segments {
		fmt.Fprintf(w, "%d\n", i+1)
		fmt.Fprintf(w, "%s --> %s\n", formatVTTTime(seg.Start), formatVTTTime(seg.End))
		fmt.Fprintf(w, "<v Speaker %d>%s\n\n", seg.Speaker+1, strings.TrimSpace(seg.Text))
	}
	return nil
}

func formatVTTTime(seconds float64) string {
	if seconds < 0 {
		seconds = 0
	}
	h := int(seconds) / 3600
	m := (int(seconds) % 3600) / 60
	s := int(seconds) % 60
	ms := int((seconds - float64(int(seconds))) * 1000)
	return fmt.Sprintf("%02d:%02d:%02d.%03d", h, m, s, ms)
}
