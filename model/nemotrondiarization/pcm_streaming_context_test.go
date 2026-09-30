package nemotrondiarization

import (
	"context"
	"errors"
	"math"
	"os"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// cancelAfterEmission cancels on request state, not a fragile count of
// context checks inside the frontend, tower, or cache.
type cancelAfterEmission struct {
	request *PCMStreamingRequest
}

func (c *cancelAfterEmission) Deadline() (time.Time, bool) { return time.Time{}, false }
func (c *cancelAfterEmission) Done() <-chan struct{}       { return nil }
func (c *cancelAfterEmission) Value(key any) any           { return nil }
func (c *cancelAfterEmission) Err() error {
	if c.request.emitted > 0 {
		return context.Canceled
	}
	return nil
}

func TestPCMStreamingRequestContextValidation(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	newRequest := func() *PCMStreamingRequest {
		t.Helper()
		s, err := LoadPCMStreamingRequest(file)
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	s := newRequest()
	if _, err := s.AppendPCMContext(nil, []float32{1}); err == nil || s.closed || s.samples != 0 {
		t.Fatal("nil context advanced stream")
	}
	if _, err := s.AppendPCMContext(context.Background(), []float32{float32(math.NaN())}); err == nil || s.closed || s.samples != 0 {
		t.Fatal("nonfinite input advanced stream")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := s.AppendPCMContext(ctx, []float32{1}); !errors.Is(err, context.Canceled) || !s.closed || s.samples != 0 {
		t.Fatalf("cancelled append err=%v closed=%v samples=%d", err, s.closed, s.samples)
	}
	if _, err := s.AppendPCM([]float32{1}); err == nil {
		t.Fatal("append after cancellation")
	}
	s = newRequest()
	if _, err := s.AppendPCMContext(context.Background(), make([]float32, 16000)); err != nil {
		t.Fatal(err)
	}
	if _, err := s.FinishContext(nil); err == nil || s.closed {
		t.Fatal("nil finish context changed request")
	}
	if _, err := s.FinishContext(ctx); !errors.Is(err, context.Canceled) || !s.closed {
		t.Fatalf("cancelled finish err=%v closed=%v", err, s.closed)
	}

	s = newRequest()
	// Cancel on the explicit post-window check. No partial logits escape.
	between := &cancelAfterEmission{request: s}
	out, err := s.AppendPCMContext(between, make([]float32, 16000*5))
	if !errors.Is(err, context.Canceled) || out != nil || !s.closed || s.emitted != lowLatencyFrames*diarizationUpsample {
		t.Fatalf("between-window cancellation err=%v out=%d closed=%v emitted=%d", err, len(out), s.closed, s.emitted)
	}
	if _, err := s.Finish(); err == nil {
		t.Fatal("accepted finish after partial cancelled request")
	}
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_PTX_REQUEST") == "1" {
		s = newRequest()
		if err := s.EnablePTXTower(); err != nil {
			t.Fatal(err)
		}
		between = &cancelAfterEmission{request: s}
		out, err = s.AppendPCMContext(between, make([]float32, 16000*5))
		if !errors.Is(err, context.Canceled) || out != nil || !s.closed || s.emitted != lowLatencyFrames*diarizationUpsample {
			t.Fatalf("PTX between-window cancellation err=%v out=%d closed=%v emitted=%d", err, len(out), s.closed, s.emitted)
		}
		if s.window.PTXTower == nil || s.window.PTXTower.tower == nil {
			t.Fatal("PTX cancellation did not exercise resident tower")
		}
		if err := s.ClosePTXTower(); err != nil {
			t.Fatal(err)
		}
		if s.window.PTXTower != nil {
			t.Fatal("PTX tower retained after cancellation cleanup")
		}
	}
}
