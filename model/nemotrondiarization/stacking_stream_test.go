package nemotrondiarization

import (
	"compress/gzip"
	"encoding/binary"
	"io"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedStackingStreamHundredSecondPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	projection, err := LoadStackingProjection(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	features := readStackingFixture(t, "../../loader/audio/testdata/nemotron_jfk_loop100_transformers_5_18_features.f32.gz", 10001*melBins)
	refFile, err := os.Open("testdata/jfk_loop100_stacking_transformers_5_18.f32.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer refFile.Close()
	gz, err := gzip.NewReader(refFile)
	if err != nil {
		t.Fatal(err)
	}
	defer gz.Close()
	for _, sizes := range [][]int{{1, 2, 7, 500, 13, 233}, {500, 500, 1}} {
		stream := &StackingStream{Projection: projection}
		var maxAbs, sumAbs float64
		var outside, total int
		var first []float32
		var firstValue float32
		check := func(got []float32) {
			if len(got) == 0 {
				return
			}
			if first == nil {
				first, firstValue = got, got[0]
			}
			bytes := make([]byte, len(got)*4)
			if _, err := io.ReadFull(gz, bytes); err != nil {
				t.Fatal(err)
			}
			for i, value := range got {
				want := math.Float32frombits(binary.LittleEndian.Uint32(bytes[i*4:]))
				delta := math.Abs(float64(value - want))
				maxAbs = math.Max(maxAbs, delta)
				sumAbs += delta
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(want)) {
					outside++
				}
			}
			total += len(got)
		}
		pos, iteration := 0, 0
		for pos < 10001 {
			n := sizes[iteration%len(sizes)]
			if n > 10001-pos {
				n = 10001 - pos
			}
			chunk := append([]float32(nil), features[pos*melBins:(pos+n)*melBins]...)
			initial := append([]float32(nil), chunk...)
			out, err := stream.AppendFeatures(chunk)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(chunk, initial) {
				t.Fatal("mutated input chunk")
			}
			check(out)
			pos += n
			iteration++
		}
		last, err := stream.Finish()
		if err != nil {
			t.Fatal(err)
		}
		check(last)
		var extra [1]byte
		if n, err := gz.Read(extra[:]); err != io.EOF || n != 0 {
			t.Fatalf("reference trailing data n=%d err=%v", n, err)
		}
		mean := sumAbs / float64(total)
		t.Logf("chunks=%d rows=%d max_abs=%g mean_abs=%g outside=%d", iteration, total/projectedWidth, maxAbs, mean, outside)
		if total != 1251*projectedWidth || first[0] != firstValue || outside != 0 || mean > 2e-5 {
			t.Fatal("streaming stacking differs from PyTorch")
		}
		if _, err := stream.AppendFeatures(features[:melBins]); err == nil {
			t.Fatal("accepted append after finish")
		}
		// Reopen the same reference for the second chunking pattern.
		if iteration > 0 && len(sizes) > 0 {
			if _, err := refFile.Seek(0, 0); err != nil {
				t.Fatal(err)
			}
			gz.Close()
			gz, err = gzip.NewReader(refFile)
			if err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestStackingStreamRejectsWithoutMutation(t *testing.T) {
	projection := &StackingProjection{weight: make([]float32, projectedWidth*stackWidth)}
	stream := &StackingStream{Projection: projection}
	if _, err := stream.AppendFeatures(nil); err == nil {
		t.Fatal("accepted empty features")
	}
	if _, err := stream.AppendFeatures(make([]float32, 501*melBins)); err == nil {
		t.Fatal("accepted oversized features")
	}
	bad := make([]float32, melBins)
	bad[0] = float32(math.NaN())
	if _, err := stream.AppendFeatures(bad); err == nil {
		t.Fatal("accepted nonfinite features")
	}
	if stream.frames != 0 {
		t.Fatal("rejected input changed state")
	}
	if _, err := stream.AppendFeatures(make([]float32, 7*melBins)); err != nil {
		t.Fatal(err)
	}
	before := stream.pending
	if _, err := stream.AppendFeatures(bad); err == nil {
		t.Fatal("accepted nonfinite after partial group")
	}
	if stream.frames != 7 || stream.pending != before {
		t.Fatal("rejected input changed pending group")
	}
	if _, err := stream.Finish(); err != nil {
		t.Fatal(err)
	}
	if _, err := stream.Finish(); err == nil {
		t.Fatal("accepted second finish")
	}
	// An exact multiple of eight has no padded output at Finish.
	exact := &StackingStream{Projection: projection}
	if out, err := exact.AppendFeatures(make([]float32, 8*melBins)); err != nil || len(out) != projectedWidth {
		t.Fatalf("exact group length=%d err=%v", len(out), err)
	}
	if out, err := exact.Finish(); err != nil || len(out) != 0 {
		t.Fatalf("exact finish length=%d err=%v", len(out), err)
	}
}
