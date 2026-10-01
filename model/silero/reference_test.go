package silero

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"testing"

	assets "github.com/rcarmo/go-pherence/loader/silero"
)

// This gate remains opt-in: ordinary tests never load a released model or
// external reference. The oracle is generated with scripts/silero-vad-reference.cpp
// linked against pinned whisper.cpp, not by the Go implementation under test.
// Reference probability tolerance must be supplied explicitly from an audited
// calibration; an absent threshold is a failure, not an automatic widening.
func TestPinnedSileroReferenceProbabilities(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_SILERO_REFERENCE")
	if path == "" {
		t.Skip("set GO_PHERENCE_SILERO_REFERENCE plus GO_PHERENCE_SILERO_MODEL, GO_PHERENCE_SILERO_PCM, GO_PHERENCE_SILERO_PCM_SHA256 and audited tolerance")
	}
	var ref struct {
		Schema         int          `json:"schema"`
		SourceRevision string       `json:"source_revision"`
		ModelSHA256    string       `json:"model_sha256"`
		Samples        int64        `json:"samples"`
		Probabilities  []float32    `json:"probabilities"`
		Segments       [][2]float64 `json:"segments_centiseconds"`
	}
	data, err := boundedReferenceFile(path, 2<<20)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) > 2<<20 || json.Unmarshal(data, &ref) != nil || ref.Schema != 1 || ref.SourceRevision != "c44b60b8053bbf2a5c1e014f11323fb3f2485177" || ref.ModelSHA256 != assets.RevisionSHA256 || ref.Samples < 1 || ref.Samples > 60*16000 || int64(len(ref.Probabilities)) != (ref.Samples+511)/512 {
		t.Fatal("invalid reference provenance/geometry")
	}
	for _, span := range ref.Segments {
		if math.IsNaN(span[0]) || math.IsNaN(span[1]) || math.IsInf(span[0], 0) || math.IsInf(span[1], 0) || span[0] < 0 || span[1] < span[0] {
			t.Fatal("invalid oracle speech span")
		}
	}
	var tolerance struct{ MaxAbsolute, MeanAbsolute float64 }
	threshold := os.Getenv("GO_PHERENCE_SILERO_TOLERANCE")
	if json.Unmarshal([]byte(threshold), &tolerance) != nil || tolerance.MaxAbsolute <= 0 || tolerance.MeanAbsolute <= 0 || math.IsNaN(tolerance.MaxAbsolute) || math.IsInf(tolerance.MaxAbsolute, 0) || math.IsNaN(tolerance.MeanAbsolute) || math.IsInf(tolerance.MeanAbsolute, 0) {
		t.Fatal("require audited GO_PHERENCE_SILERO_TOLERANCE JSON MaxAbsolute/MeanAbsolute")
	}
	file, err := assets.LoadPath(context.Background(), os.Getenv("GO_PHERENCE_SILERO_MODEL"), assets.RevisionSHA256)
	if err != nil {
		t.Fatal(err)
	}
	model, err := New(file)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := boundedReferenceFile(os.Getenv("GO_PHERENCE_SILERO_PCM"), 60*16000*4)
	if err != nil || int64(len(raw)) != ref.Samples*4 {
		t.Fatal("PCM footprint", err)
	}
	pin := os.Getenv("GO_PHERENCE_SILERO_PCM_SHA256")
	digest := sha256.Sum256(raw)
	if len(pin) != 64 || hex.EncodeToString(digest[:]) != pin {
		t.Fatal("reference PCM SHA256 mismatch")
	}
	pcm := make([]float32, ref.Samples)
	for i := range pcm {
		pcm[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	scores, err := model.Probabilities(context.Background(), &memoryPCM{data: pcm}, ref.Samples)
	if err != nil || len(scores) != len(ref.Probabilities) {
		t.Fatal("probabilities", err)
	}
	// Trained fresh/reset/retry: an interrupted frame cannot advance the LSTM.
	stream, err := model.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	var firstFrame, secondFrame [512]float32
	copy(firstFrame[:], pcm)
	if len(pcm) > 512 {
		copy(secondFrame[:], pcm[512:])
	}
	first, err := stream.Probability(context.Background(), firstFrame[:])
	if err != nil || first != scores[0] {
		t.Fatal("trained first frame", first, err)
	}
	if len(scores) > 1 {
		if _, err := stream.Probability(&cancelAtCheck{at: 7}, secondFrame[:]); !errors.Is(err, context.Canceled) {
			t.Fatal("trained final cancellation", err)
		}
		second, err := stream.Probability(context.Background(), secondFrame[:])
		if err != nil || second != scores[1] {
			t.Fatal("trained cancelled retry", second, err)
		}
	}
	stream.Reset()
	reset, err := stream.Probability(context.Background(), firstFrame[:])
	if err != nil || reset != first {
		t.Fatal("trained reset", reset, err)
	}
	var max, sum float64
	var flips, endFlips int
	for i, p := range scores {
		reference := ref.Probabilities[i]
		if !finite(reference) || reference < 0 || reference > 1 {
			t.Fatal("invalid oracle score")
		}
		d := math.Abs(float64(p - reference))
		max = math.Max(max, d)
		sum += d
		if (p >= 0.5) != (reference >= 0.5) {
			flips++
		}
		if (p < float32(0.5)-float32(0.15)) != (reference < float32(0.5)-float32(0.15)) {
			endFlips++
		}
	}
	mean := sum / float64(len(scores))
	t.Logf("scores=%d max_abs=%g mean_abs=%g threshold_flips=%d end_threshold_flips=%d", len(scores), max, mean, flips, endFlips)
	if max > tolerance.MaxAbsolute || mean > tolerance.MeanAbsolute || flips != 0 || endFlips != 0 {
		t.Fatal("independent probability parity failed")
	}
	// whisper.cpp samples_to_cs rounds to nearest centisecond (ties up),
	// not down. Its EOF uses padded frame length; the native API deliberately
	// clips to actual PCM. Clip the oracle's EOF to the same real audio bound.
	// Exact sample spans from native and oracle probabilities must also agree.
	referenceSpans, err := SpeechSpans(ref.Probabilities, ref.Samples, DefaultSegmentOptions())
	if err != nil {
		t.Fatal(err)
	}
	spans, err := SpeechSpans(scores, ref.Samples, DefaultSegmentOptions())
	if err != nil {
		t.Fatal(err)
	}
	if len(spans) != len(ref.Segments) || len(spans) != len(referenceSpans) {
		t.Fatalf("speech span count=%d reference=%d reference_scores=%d", len(spans), len(ref.Segments), len(referenceSpans))
	}
	for i, s := range spans {
		if s != referenceSpans[i] {
			t.Fatalf("span%d exact samples=%v reference_scores=%v", i, s, referenceSpans[i])
		}
		for k, v := range []int64{s.Start, s.End} {
			got := float64(referenceCentiseconds(v))
			want := min(ref.Segments[i][k], float64(referenceCentiseconds(ref.Samples)))
			if got != want {
				t.Fatalf("span%d boundary%d centiseconds=%g reference=%g", i, k, got, want)
			}
		}
	}
}

// Matches positive-sample samples_to_cs in the pinned original.
func referenceCentiseconds(samples int64) int64 { return (samples + 80) / 160 }

func TestReferenceCentisecondsUsesNearestNotFloor(t *testing.T) {
	for _, tc := range []struct{ samples, want int64 }{{0, 0}, {79, 0}, {80, 1}, {159, 1}, {160, 1}, {52256, 327}} {
		if got := referenceCentiseconds(tc.samples); got != tc.want {
			t.Fatalf("samples=%d got=%d want=%d", tc.samples, got, tc.want)
		}
	}
}

func boundedReferenceFile(path string, limit int64) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	err = errors.Join(err, f.Close())
	if err != nil {
		return nil, err
	}
	if int64(len(b)) > limit {
		return nil, fmt.Errorf("reference fixture too large")
	}
	return b, nil
}
