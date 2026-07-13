package mosstranscribe

import (
	"strings"
	"testing"
)

func TestAudioSpanIDsTimeMarkers(t *testing.T) {
	var digits [10]int
	for i := range digits {
		digits[i] = 100 + i
	}
	ids := AudioSpanIDs(375, 999, digits, 5)
	placeholderCount := 0
	var markers []int
	for _, id := range ids {
		if id == 999 {
			placeholderCount++
		} else {
			markers = append(markers, id)
		}
	}
	if placeholderCount != 375 {
		t.Fatalf("placeholders=%d want 375", placeholderCount)
	}
	// 30 seconds includes markers 5,10,15,20,25,30.
	want := []int{105, 101, 100, 101, 105, 102, 100, 102, 105, 103, 100}
	if len(markers) != len(want) {
		t.Fatalf("markers=%v want %v", markers, want)
	}
	for i := range want {
		if markers[i] != want[i] {
			t.Fatalf("markers=%v want %v", markers, want)
		}
	}
	if ids[62] != 105 { // floor(12.5*5)=62 placeholders precede the marker.
		t.Fatalf("first marker position/value=%d want 105", ids[62])
	}
}

func TestInsertAudioEmbeddings(t *testing.T) {
	ids := []int{1, 9, 2, 9}
	tokens := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	audio := []float32{10, 11, 20, 21}
	out := make([]float32, len(tokens))
	if err := InsertAudioEmbeddingsTo(out, tokens, audio, ids, 9, 2); err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 2, 10, 11, 5, 6, 20, 21}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("out=%v want %v", out, want)
		}
	}
	bad := make([]float32, len(tokens))
	for i := range bad {
		bad[i] = -1
	}
	if err := InsertAudioEmbeddingsTo(bad, tokens, audio[:2], ids, 9, 2); err == nil || !strings.Contains(err.Error(), "placeholders") {
		t.Fatalf("count mismatch error=%v", err)
	}
	for _, value := range bad {
		if value != -1 {
			t.Fatal("count mismatch partially wrote output")
		}
	}
}
