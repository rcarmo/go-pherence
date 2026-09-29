package nemotrondiarization

import (
	"math"
	"reflect"
	"testing"
)

func TestSegmentStreamMatchesExtractSegments(t *testing.T) {
	logits := make([]float32, 37*diarizationSpeakers)
	for frame := 0; frame < 37; frame++ {
		for speaker := 0; speaker < diarizationSpeakers; speaker++ {
			logits[frame*diarizationSpeakers+speaker] = -2
		}
		if frame < 4 || frame >= 9 && frame < 25 {
			logits[frame*diarizationSpeakers] = 3
		}
		if frame >= 20 {
			logits[frame*diarizationSpeakers+1] = 1
		}
	}
	want, err := ExtractSegments(logits, makeTrueMask(37))
	if err != nil {
		t.Fatal(err)
	}
	for _, sizes := range [][]int{{1}, {4, 7, 11, 15}, {37}} {
		var s SegmentStream
		var got []Segment
		for offset, step := 0, 0; offset < 37; step++ {
			n := sizes[step%len(sizes)]
			if n > 37-offset {
				n = 37 - offset
			}
			part, err := s.Append(logits[offset*diarizationSpeakers : (offset+n)*diarizationSpeakers])
			if err != nil {
				t.Fatal(err)
			}
			got = append(got, part...)
			offset += n
		}
		last, err := s.Finish()
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, last...)
		// Returned completed spans are chronological across calls; the
		// offline extractor sorts the whole recording by start timestamp.
		sortSegments(got)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("sizes=%v got=%v want=%v", sizes, got, want)
		}
		if _, err := s.Append(logits[:8]); err == nil {
			t.Fatal("accepted append after finish")
		}
	}
}

func makeTrueMask(n int) []bool {
	mask := make([]bool, n)
	for i := range mask {
		mask[i] = true
	}
	return mask
}

func TestSegmentStreamRejectsInvalidWithoutAdvance(t *testing.T) {
	var s SegmentStream
	for _, bad := range [][]float32{nil, make([]float32, 7), {float32(math.NaN()), 0, 0, 0, 0, 0, 0, 0}} {
		if _, err := s.Append(bad); err == nil || s.frames != 0 || s.starts[0] != 0 {
			t.Fatal("invalid logits advanced segment stream")
		}
	}
	if _, err := s.Finish(); err == nil {
		t.Fatal("accepted empty finish")
	}
	good := make([]float32, 8)
	good[0] = 1
	if _, err := s.Append(good); err != nil {
		t.Fatal(err)
	}
	part, err := s.Finish()
	if err != nil || !reflect.DeepEqual(part, []Segment{{Start: 0, End: 0.01, Speaker: 0}}) {
		t.Fatalf("terminal segment=%v err=%v", part, err)
	}
}
