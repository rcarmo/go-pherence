package audio

import (
	"compress/gzip"
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestNemotronMelStreamJFKPyTorchParity(t *testing.T) {
	pcm := nemotronJFK(t)
	file, err := os.Open("testdata/nemotron_jfk_transformers_5_18_features.f32.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	gz, err := gzip.NewReader(file)
	if err != nil {
		t.Fatal(err)
	}
	defer gz.Close()
	data, err := io.ReadAll(io.LimitReader(gz, 1101*128*4+1))
	if err != nil || len(data) != 1101*128*4 {
		t.Fatalf("reference length=%d err=%v", len(data), err)
	}
	for _, sizes := range [][]int{{1, 159, 1, 7, 80000, 150, 80000, 534}, {80000, 80000, 16000}} {
		var stream NemotronMelStream
		var result []float32
		var first []float32
		var firstValue float32
		pos, iteration := 0, 0
		for pos < len(pcm) {
			n := sizes[iteration%len(sizes)]
			if n > len(pcm)-pos {
				n = len(pcm) - pos
			}
			input := append([]float32(nil), pcm[pos:pos+n]...)
			original := append([]float32(nil), input...)
			part, err := stream.AppendPCM(input)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(input, original) {
				t.Fatal("mutated input chunk")
			}
			if len(part) > 0 && first == nil {
				first = part
				firstValue = part[0]
			}
			result = append(result, part...)
			pos += n
			iteration++
		}
		last, err := stream.Finish()
		if err != nil {
			t.Fatal(err)
		}
		result = append(result, last...)
		if len(result) != 1101*128 || first[0] != firstValue {
			t.Fatalf("unexpected output length=%d or mutated prior return", len(result))
		}
		var maxAbs, sumAbs float64
		var outside int
		for i, value := range result {
			want := math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
			delta := math.Abs(float64(value - want))
			maxAbs = math.Max(maxAbs, delta)
			sumAbs += delta
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 5e-4+1e-5*math.Abs(float64(want)) {
				outside++
			}
		}
		t.Logf("chunks=%d max_abs=%g mean_abs=%g outside=%d", iteration, maxAbs, sumAbs/float64(len(result)), outside)
		if outside != 0 || sumAbs/float64(len(result)) > 1e-5 {
			t.Fatal("streaming mel differs from pinned PyTorch")
		}
	}
}

func TestNemotronMelStreamLongBoundedState(t *testing.T) {
	var stream NemotronMelStream
	chunk := make([]float32, nemotronStreamMaxChunk)
	var total int64
	for i := 0; i < 20; i++ { // 100 seconds, beyond the offline 30-second limit
		part, err := stream.AppendPCM(chunk)
		if err != nil {
			t.Fatal(err)
		}
		if len(part) > 502*128 {
			t.Fatal("unbounded returned chunk")
		}
		total += int64(len(part)) / 128
	}
	last, err := stream.Finish()
	if err != nil {
		t.Fatal(err)
	}
	total += int64(len(last)) / 128
	if total != 10001 || stream.samples != 1600000 || stream.nextFrame != 10000 {
		t.Fatalf("rows=%d samples=%d next=%d", total, stream.samples, stream.nextFrame)
	}
	if _, err := stream.AppendPCM(chunk[:1]); err == nil {
		t.Fatal("accepted append after finish")
	}
	if _, err := stream.Finish(); err == nil {
		t.Fatal("accepted second finish")
	}
}

func TestNemotronMelStreamHundredSecondPyTorchParity(t *testing.T) {
	file, err := os.Open(filepath.Join("testdata", "nemotron_jfk_loop100_transformers_5_18_features.f32.gz"))
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	gz, err := gzip.NewReader(file)
	if err != nil {
		t.Fatal(err)
	}
	defer gz.Close()
	var stream NemotronMelStream
	pcm := nemotronJFK(t)
	var maxAbs, sumAbs float64
	var outside, elements int
	compare := func(part []float32) {
		bytes := make([]byte, 4*len(part))
		if _, err := io.ReadFull(gz, bytes); err != nil {
			t.Fatal(err)
		}
		for i, value := range part {
			want := math.Float32frombits(binary.LittleEndian.Uint32(bytes[i*4:]))
			delta := math.Abs(float64(value - want))
			maxAbs = math.Max(maxAbs, delta)
			sumAbs += delta
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 5e-4+1e-5*math.Abs(float64(want)) {
				outside++
			}
		}
		elements += len(part)
	}
	for i := 0; i < 20; i++ {
		chunk := make([]float32, nemotronStreamMaxChunk)
		for j := range chunk {
			chunk[j] = pcm[(i*nemotronStreamMaxChunk+j)%len(pcm)]
		}
		part, err := stream.AppendPCM(chunk)
		if err != nil {
			t.Fatal(err)
		}
		compare(part)
	}
	last, err := stream.Finish()
	if err != nil {
		t.Fatal(err)
	}
	compare(last)
	var extra [1]byte
	if n, err := gz.Read(extra[:]); err != io.EOF || n != 0 {
		t.Fatalf("reference has trailing data n=%d err=%v", n, err)
	}
	mean := sumAbs / float64(elements)
	t.Logf("100s rows=%d max_abs=%g mean_abs=%g outside=%d", elements/128, maxAbs, mean, outside)
	if elements != 10001*128 || outside != 0 || mean > 1e-5 {
		t.Fatal("long streaming mel differs from independent PyTorch")
	}
}

func TestNemotronMelStreamRejectsWithoutMutation(t *testing.T) {
	var stream NemotronMelStream
	if _, err := stream.Finish(); err == nil {
		t.Fatal("accepted empty stream")
	}
	if _, err := stream.AppendPCM(make([]float32, nemotronStreamMaxChunk+1)); err == nil {
		t.Fatal("accepted oversized chunk")
	}
	if stream.samples != 0 {
		t.Fatal("oversized input changed stream")
	}
	if _, err := stream.AppendPCM([]float32{1, 2, float32(math.NaN())}); err == nil {
		t.Fatal("accepted nonfinite chunk")
	}
	if stream.samples != 0 {
		t.Fatal("nonfinite input changed stream")
	}
	stream.samples = math.MaxInt64 - 161
	if _, err := stream.AppendPCM(make([]float32, 2)); err == nil {
		t.Fatal("accepted overflowing frame count")
	}
	if stream.samples != math.MaxInt64-161 {
		t.Fatal("overflow rejection changed stream")
	}
	stream.samples = 0
	if _, err := stream.AppendPCM([]float32{1}); err != nil {
		t.Fatal(err)
	}
	last, err := stream.Finish()
	if err != nil || len(last) != 128 {
		t.Fatalf("single-sample finish length=%d err=%v", len(last), err)
	}
}
