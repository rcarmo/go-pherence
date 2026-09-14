package omnivoice

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// Decoder-only: model load, Prepare and output allocation precede timing.
// Codes match scripts/omnivoice-upstream-codec-benchmark.py.
func BenchmarkRealCodecDecode75Frames(b *testing.B) {
	path := os.Getenv("GO_PHERENCE_REAL_OMNIVOICE")
	if path == "" {
		b.Skip("set GO_PHERENCE_REAL_OMNIVOICE")
	}
	old := runtime.GOMAXPROCS(2)
	defer runtime.GOMAXPROCS(old)
	w, err := loader.LoadCodecDecoder(filepath.Join(path, "audio_tokenizer"))
	if err != nil {
		b.Fatal(err)
	}
	d, err := NewCodecDecoder(w)
	if err != nil {
		b.Fatal(err)
	}
	const books, frames = 8, 75
	if err := d.Prepare(frames); err != nil {
		b.Fatal(err)
	}
	codes := make([]int, books*frames)
	for book := 0; book < books; book++ {
		for t := 0; t < frames; t++ {
			codes[book*frames+t] = ((book*frames + t) * 13) % 1024
		}
	}
	out := make([]float32, frames*960)
	// One untimed warm-up in both runtimes.
	if err := d.DecodeInto(context.Background(), out, codes, books, frames); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := d.DecodeInto(context.Background(), out, codes, books, frames); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	hash := sha256.New()
	var raw [4]byte
	for _, v := range out {
		binary.LittleEndian.PutUint32(raw[:], math.Float32bits(v))
		hash.Write(raw[:])
	}
	b.Logf("waveform_float32_sha256=%x", hash.Sum(nil))
	for _, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			b.Fatal("nonfinite waveform")
		}
	}
}
