package main

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/model/nemotrondiarization"
)

func TestSortSegmentsOverlappingSpeakers(t *testing.T) {
	segments := []nemotrondiarization.Segment{
		{Start: 1, End: 2, Speaker: 1},
		{Start: 1, End: 3, Speaker: 0},
		{Start: .5, End: 4, Speaker: 2},
	}
	sortSegments(segments)
	for i, want := range []int{2, 0, 1} {
		if segments[i].Speaker != want {
			t.Fatalf("speaker at %d=%d want=%d", i, segments[i].Speaker, want)
		}
	}
}

func TestRunRejectsInvalidFlagsAndNonMonoWAV(t *testing.T) {
	var stdout, stderr bytes.Buffer
	for _, args := range [][]string{
		{"-task", "asr", "-backend", "unknown", "-input", "in.wav", "-model", "model.safetensors"},
		{"-task", "diarization", "-backend", "simd", "-input", "in.wav", "-model", "model.safetensors", "-tokenizer", "tokens.json"},
		{"-task", "asr", "-input", "in.mp3", "-model", "model.safetensors"},
		{"-task", "asr", "-backend", "vulkan", "-vulkan-tower", "-input", "in.wav", "-model", "model.safetensors"},
		{"-task", "diarization", "-backend", "ptx", "-vulkan-tower", "-input", "in.wav", "-model", "model.safetensors"},
		{"-task", "diarization", "-backend", "simd", "-vulkan-tower", "-input", "in.wav", "-model", "model.safetensors"},
	} {
		stdout.Reset()
		stderr.Reset()
		if err := run(args, &stdout, &stderr); err == nil || stdout.Len() != 0 {
			t.Fatalf("accepted args=%v stdout=%q", args, stdout.String())
		}
	}
	path := filepath.Join(t.TempDir(), "stereo.wav")
	header := make([]byte, 44)
	copy(header[:4], "RIFF")
	copy(header[8:12], "WAVE")
	copy(header[12:16], "fmt ")
	binary.LittleEndian.PutUint32(header[16:20], 16)
	binary.LittleEndian.PutUint16(header[20:22], 1)
	binary.LittleEndian.PutUint16(header[22:24], 2)
	if err := os.WriteFile(path, header, 0600); err != nil {
		t.Fatal(err)
	}
	if err := requireMonoWAV(path); err == nil || !strings.Contains(err.Error(), "mono") {
		t.Fatalf("stereo WAV error=%v", err)
	}
}
