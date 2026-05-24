package speaker

import (
	"testing"

	"github.com/rcarmo/go-pherence/models/whisper"
)

func TestAlignSpeakers(t *testing.T) {
	whisperSegs := []whisper.Segment{
		{Start: 0.0, End: 2.0, Text: "Hello there"},
		{Start: 2.5, End: 4.0, Text: "How are you"},
		{Start: 5.0, End: 7.0, Text: "I am fine"},
	}

	vadSegs := []VADSegment{
		{Start: 0.0, End: 2.2}, // speaker A
		{Start: 2.3, End: 4.5}, // speaker B
		{Start: 4.8, End: 7.2}, // speaker A
	}

	labels := []int{0, 1, 0}

	result := AlignSpeakers(whisperSegs, vadSegs, labels)
	if len(result) != 3 {
		t.Fatalf("result length=%d want 3", len(result))
	}

	// Segment 0 overlaps most with VAD 0 (speaker 0)
	if result[0].Speaker != 0 {
		t.Fatalf("segment 0 speaker=%d want 0", result[0].Speaker)
	}
	// Segment 1 overlaps most with VAD 1 (speaker 1)
	if result[1].Speaker != 1 {
		t.Fatalf("segment 1 speaker=%d want 1", result[1].Speaker)
	}
	// Segment 2 overlaps most with VAD 2 (speaker 0)
	if result[2].Speaker != 0 {
		t.Fatalf("segment 2 speaker=%d want 0", result[2].Speaker)
	}
}

func TestOverlapDuration(t *testing.T) {
	// Full overlap
	if o := overlapDuration(1, 3, 0, 5); o != 2 {
		t.Fatalf("full overlap=%f want 2", o)
	}
	// Partial overlap
	if o := overlapDuration(1, 3, 2, 5); o != 1 {
		t.Fatalf("partial overlap=%f want 1", o)
	}
	// No overlap
	if o := overlapDuration(1, 2, 3, 4); o != 0 {
		t.Fatalf("no overlap=%f want 0", o)
	}
}
