package media_test

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio/media"
)

func ExampleNewGo264() {
	backend, err := media.NewGo264(media.Go264Config{})
	if err != nil {
		panic(err)
	}
	var adapter media.Adapter = backend
	fmt.Println(adapter != nil)
	// Output: true
}

// This external-package test uses only the supported public library surface.
// It needs no FFmpeg, model weights, network, build tags or internal helpers.
func TestGo264ExternalImportDecodeAndRead(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	src, dst := filepath.Join(dir, "input.wav"), filepath.Join(dir, "canonical.wav")
	samples := []int16{-32768, -123, 0, 123, 32767}
	wav := make([]byte, 44+2*len(samples))
	copy(wav[0:4], "RIFF")
	binary.LittleEndian.PutUint32(wav[4:8], uint32(len(wav)-8))
	copy(wav[8:16], "WAVEfmt ")
	binary.LittleEndian.PutUint32(wav[16:20], 16)
	binary.LittleEndian.PutUint16(wav[20:22], 1)
	binary.LittleEndian.PutUint16(wav[22:24], 1)
	binary.LittleEndian.PutUint32(wav[24:28], 16000)
	binary.LittleEndian.PutUint32(wav[28:32], 32000)
	binary.LittleEndian.PutUint16(wav[32:34], 2)
	binary.LittleEndian.PutUint16(wav[34:36], 16)
	copy(wav[36:40], "data")
	binary.LittleEndian.PutUint32(wav[40:44], uint32(2*len(samples)))
	for i, sample := range samples {
		binary.LittleEndian.PutUint16(wav[44+2*i:], uint16(sample))
	}
	if err := os.WriteFile(src, wav, 0o600); err != nil {
		t.Fatal(err)
	}
	backend, err := media.NewGo264(media.Go264Config{})
	if err != nil {
		t.Fatal(err)
	}
	var adapter media.Adapter = backend
	probe, err := adapter.Probe(ctx, src)
	if err != nil || probe.Timeline.Samples != media.SampleCount(len(samples)) {
		t.Fatalf("probe: %+v, %v", probe, err)
	}
	result, err := adapter.DecodeToFile(ctx, src, dst)
	if err != nil {
		t.Fatal(err)
	}
	if result.Timeline.Samples != media.SampleCount(len(samples)) || result.Format.SampleRate != media.CanonicalSampleRate {
		t.Fatalf("decode: %+v", result)
	}
	reader, err := media.OpenCanonicalPCM(ctx, result.Path)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	got := make([]float32, len(samples))
	n, err := reader.ReadSamplesAt(ctx, got, 0)
	if err != nil || n != len(samples) {
		t.Fatalf("read: n=%d, err=%v", n, err)
	}
	for i, want := range samples {
		if got[i] != float32(want)/32768 {
			t.Fatalf("sample %d: got %g, want %g", i, got[i], float32(want)/32768)
		}
	}
	if _, err := os.Stat(src); err != nil {
		t.Fatalf("input was not retained: %v", err)
	}
}
