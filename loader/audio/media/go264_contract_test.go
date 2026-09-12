package media

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// deterministicCancelContext cancels at a known number of boundary checks;
// this avoids timing races and tests cancellation after temporary output exists.
type deterministicCancelContext struct {
	context.Context
	checks, after int
	done          chan struct{}
}

func (c *deterministicCancelContext) Err() error {
	c.checks++
	if c.checks >= c.after {
		select {
		case <-c.done:
		default:
			close(c.done)
		}
		return context.Canceled
	}
	return nil
}
func (c *deterministicCancelContext) Done() <-chan struct{} { return c.done }
func TestGo264MidDecodeCancelCleanup(t *testing.T) {
	for _, after := range []int{15, 40, 100} {
		dir := t.TempDir()
		src, dst := filepath.Join(dir, "source.wav"), filepath.Join(dir, "out.wav")
		writeSyntheticWAV(t, src, 48000, 96000)
		g := newTestGo264(t, Go264Config{})
		ctx := &deterministicCancelContext{Context: context.Background(), after: after, done: make(chan struct{})}
		if _, e := g.DecodeToFile(ctx, src, dst); !errors.Is(e, context.Canceled) {
			t.Fatal(after, e)
		}
		files, _ := os.ReadDir(dir)
		if len(files) != 1 || files[0].Name() != "source.wav" {
			t.Fatal("cancel publication/leak", files)
		}
	}
}
func TestGo264ConfigAndCanonicalReaderBoundary(t *testing.T) {
	for _, cfg := range []Go264Config{{MaxInputBytes: -1}, {MaxDuration: 5 * time.Hour}, {MaxDecodeOutputBytes: -1}} {
		if _, e := NewGo264(cfg); e == nil {
			t.Fatal(cfg)
		}
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "source.wav")
	writeSyntheticWAV(t, src, 16000, 1)
	g := newTestGo264(t, Go264Config{})
	if _, e := g.DecodeToFile(context.Background(), src, src); e == nil {
		t.Fatal("self clobber")
	}
	if _, e := g.Probe(nil, src); e == nil {
		t.Fatal("nil context")
	}
}
