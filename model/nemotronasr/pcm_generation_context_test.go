package nemotronasr

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/audio"
)

// This test context flips only after the first completed greedy chunk. It
// avoids coupling cancellation to the number of checks inside GPU/CPU kernels.
type cancelAfterASRChunk struct{ stream *PCMGenerationStream }

func (c *cancelAfterASRChunk) Deadline() (time.Time, bool) { return time.Time{}, false }
func (c *cancelAfterASRChunk) Done() <-chan struct{}       { return nil }
func (c *cancelAfterASRChunk) Value(any) any               { return nil }
func (c *cancelAfterASRChunk) Err() error {
	if c.stream.greedy.frames > 0 {
		return context.Canceled
	}
	return nil
}

func TestReleasedPCMGenerationCancelAfterChunk(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_ASR_CANCEL") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_ASR_CANCEL=1")
	}
	model := releasedPCMGenerationModel(t)
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 || len(pcm) < 80000 {
		t.Fatalf("JFK PCM rate=%d len=%d err=%v", rate, len(pcm), err)
	}
	for _, backend := range []string{"simd", "ptx", "vulkan"} {
		t.Run(backend, func(t *testing.T) {
			s := &PCMGenerationStream{Model: model}
			var projector *DeviceSubsamplingProjector
			if backend != "simd" {
				projector = &DeviceSubsamplingProjector{Backend: backend}
				s.Projector = projector
				defer projector.Close()
			}
			var before ptx.Stats
			if backend == "ptx" {
				previous := ptx.SetStatsEnabled(true)
				defer ptx.SetStatsEnabled(previous)
				before = ptx.StatsSnapshot()
			}
			ctx := &cancelAfterASRChunk{stream: s}
			tokens, frames, err := s.AppendPCM(ctx, pcm[:80000])
			if !errors.Is(err, context.Canceled) || tokens != nil || frames != nil || !s.closed || s.greedy.frames == 0 {
				t.Fatalf("cancelled generation err=%v tokens=%d frames=%d closed=%v completed=%d", err, len(tokens), len(frames), s.closed, s.greedy.frames)
			}
			if _, _, err := s.Finish(context.Background()); err == nil {
				t.Fatal("accepted finish after cancellation")
			}
			if projector != nil {
				if err := projector.Close(); err != nil {
					t.Fatal(err)
				}
				if !projector.closed {
					t.Fatal("projector not closed")
				}
			}
			if backend == "ptx" {
				after := ptx.StatsSnapshot()
				if after.Mallocs-before.Mallocs != after.Frees-before.Frees || after.MallocBytes-before.MallocBytes != after.FreeBytes-before.FreeBytes {
					t.Fatalf("PTX projector leaked: before=%+v after=%+v", before, after)
				}
			}
		})
	}
}
