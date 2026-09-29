package nemotronasr

import (
	"compress/gzip"
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio"
)

func TestASRMelChunkStreamJFK(t *testing.T) {
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 || len(pcm) != 176000 {
		t.Fatalf("JFK WAV rate=%d length=%d err=%v", rate, len(pcm), err)
	}
	refFile, err := os.Open(filepath.Join("..", "..", "loader", "audio", "testdata", "nemotron_jfk_transformers_5_18_features.f32.gz"))
	if err != nil {
		t.Fatal(err)
	}
	defer refFile.Close()
	gz, err := gzip.NewReader(refFile)
	if err != nil {
		t.Fatal(err)
	}
	defer gz.Close()
	data, err := io.ReadAll(gz)
	if err != nil || len(data) != 1101*128*4 {
		t.Fatalf("reference bytes=%d err=%v", len(data), err)
	}
	var s ASRMelChunkStream
	var chunks []ASRMelChunk
	for pos, i := 0, 0; pos < len(pcm); i++ {
		n := []int{1, 7, 4040, 5520, 733, 80000}[i%6]
		if n > len(pcm)-pos {
			n = len(pcm) - pos
		}
		in := append([]float32(nil), pcm[pos:pos+n]...)
		before := append([]float32(nil), in...)
		part, err := s.AppendPCM(in)
		if err != nil || !reflect.DeepEqual(in, before) {
			t.Fatalf("PCM append or ownership at %d: %v", pos, err)
		}
		chunks = append(chunks, part...)
		pos += n
	}
	last, err := s.Finish()
	if err != nil {
		t.Fatal(err)
	}
	chunks = append(chunks, last...)
	if len(s.pending) != 0 || !s.closed || len(chunks) == 0 {
		t.Fatalf("unflushed/empty scheduler: chunks=%d pending=%d", len(chunks), len(s.pending))
	}
	row := 0
	var maxAbs float64
	for i, chunk := range chunks {
		if chunk.Frames != len(chunk.Features)/128 || (i == 0 && chunk.Frames != 26) || (i != 0 && chunk.Frames != 32) || chunk.Valid < 1 || chunk.Valid > chunk.Frames {
			t.Fatalf("chunk %d geometry frames=%d valid=%d", i, chunk.Frames, chunk.Valid)
		}
		for j := 0; j < chunk.Valid*128; j++ {
			want := math.Float32frombits(binary.LittleEndian.Uint32(data[(row*128+j)*4:]))
			diff := math.Abs(float64(chunk.Features[j] - want))
			maxAbs = math.Max(maxAbs, diff)
			if diff > 5e-4+1e-5*math.Abs(float64(want)) {
				t.Fatalf("mel row=%d col=%d delta=%g", row+j/128, j%128, diff)
			}
		}
		for _, v := range chunk.Features[chunk.Valid*128:] {
			if v != 0 {
				t.Fatalf("chunk %d nonzero masked tail", i)
			}
		}
		row += chunk.Valid
	}
	if row != 1100 {
		t.Fatalf("valid rows=%d", row)
	}
	t.Logf("chunks=%d valid_rows=%d max_abs=%g", len(chunks), row, maxAbs)
	if _, err := s.AppendPCM(pcm[:1]); err == nil {
		t.Fatal("append after finish")
	}
	if _, err := s.Finish(); err == nil {
		t.Fatal("second finish")
	}
}

func TestASRMelChunkStreamHundredSecondBounded(t *testing.T) {
	var s ASRMelChunkStream
	pcm := make([]float32, 80000)
	validRows, chunks := 0, 0
	for i := 0; i < 20; i++ {
		part, err := s.AppendPCM(pcm)
		if err != nil {
			t.Fatal(err)
		}
		for _, chunk := range part {
			validRows += chunk.Valid
			chunks++
		}
		if len(s.pending) >= 32*128 || cap(s.pending) > 64*128 {
			t.Fatalf("unbounded mel pending: len=%d cap=%d", len(s.pending), cap(s.pending))
		}
	}
	last, err := s.Finish()
	if err != nil {
		t.Fatal(err)
	}
	for _, chunk := range last {
		validRows += chunk.Valid
		chunks++
	}
	if validRows != 10000 || chunks != 313 {
		t.Fatalf("100-second valid=%d chunks=%d", validRows, chunks)
	}
}

func TestASRMelChunkStreamRejectsInvalidInput(t *testing.T) {
	var s ASRMelChunkStream
	for _, in := range [][]float32{nil, make([]float32, 80001), {float32(math.NaN())}} {
		if _, err := s.AppendPCM(in); err == nil || s.first || len(s.pending) != 0 {
			t.Fatal("invalid input advanced scheduler")
		}
	}
	if _, err := s.Finish(); err == nil || s.closed {
		t.Fatal("empty finish accepted")
	}
	if _, err := s.AppendPCM(make([]float32, 320)); err != nil {
		t.Fatal(err)
	}
	chunks, err := s.Finish()
	if err != nil || len(chunks) != 1 {
		t.Fatalf("short terminal chunks=%d err=%v", len(chunks), err)
	}
	if chunks[0].Frames != 26 || chunks[0].Valid != 2 {
		t.Fatalf("short terminal frames=%d valid=%d", chunks[0].Frames, chunks[0].Valid)
	}
}
