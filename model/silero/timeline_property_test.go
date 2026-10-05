package silero

import (
	"context"
	"io"
	"math/rand"
	"testing"
)

// Independent explicit retained-index list is the timeline oracle. It does
// not call TimeMap to generate expected values and covers awkward read offsets,
// tails and equal boundaries across many deterministic span/gap shapes.
func TestCompactedTimelineAgainstExplicitOriginalIndices(t *testing.T) {
	rng := rand.New(rand.NewSource(0x51264))
	for trial := 0; trial < 40; trial++ {
		data := make([]float32, 512)
		for i := range data {
			data[i] = float32(i)
		}
		var spans []Span
		var indices []int64
		cursor := int64(0)
		for cursor < 470 {
			cursor += int64(rng.Intn(12) + 1)
			end := cursor + int64(rng.Intn(16)+1)
			if end > 512 {
				end = 512
			}
			spans = append(spans, Span{Start: cursor, End: end})
			for i := cursor; i < end; i++ {
				indices = append(indices, i)
			}
			cursor = end
		}
		reader, err := NewCompactedReader(&memoryPCM{data: data}, spans, 512)
		if err != nil {
			t.Fatal(err)
		}
		if reader.Samples() != int64(len(indices)) {
			t.Fatal("index count")
		}
		for start := 0; start < len(indices); start += rng.Intn(7) + 1 {
			width := rng.Intn(31) + 1
			dst := make([]float32, width)
			n, err := reader.ReadSamplesAt(context.Background(), dst, int64(start))
			wantN := width
			if wantN > len(indices)-start {
				wantN = len(indices) - start
			}
			if n != wantN || (wantN < width && err != io.EOF) || (wantN == width && err != nil) {
				t.Fatal("read footprint", trial, start, n, wantN, err)
			}
			for i := 0; i < n; i++ {
				if dst[i] != float32(indices[start+i]) {
					t.Fatal("retained read index", trial, start, i, dst[i], indices[start+i])
				}
			}
			mapped, err := reader.MapStart(int64(start))
			if err != nil || mapped != indices[start] {
				t.Fatal("start index", trial, start, mapped, err)
			}
			if start > 0 {
				mapped, err := reader.MapEnd(int64(start))
				if err != nil || mapped != indices[start-1]+1 {
					t.Fatal("end index", trial, start, mapped, err)
				}
			}
		}
		mapped, err := reader.MapEnd(int64(len(indices)))
		if err != nil || mapped != indices[len(indices)-1]+1 {
			t.Fatal("final boundary")
		}
	}
}
