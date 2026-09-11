package media

import (
	"bytes"
	"context"
	"errors"
	"os/exec"
	"time"
)

var errWriterLimit = errors.New("command output limit exceeded")

type execRunner struct{}

func (execRunner) Run(ctx context.Context, cmd Command) error {
	c := exec.CommandContext(ctx, cmd.Path, cmd.Args...)
	c.WaitDelay = 2 * time.Second
	c.Stdout = cmd.Stdout
	c.Stderr = cmd.Stderr
	return c.Run()
}

type hardLimitBuffer struct {
	buf   bytes.Buffer
	limit int64
}

func (b *hardLimitBuffer) Write(p []byte) (int, error) {
	if b.limit <= 0 {
		return len(p), nil
	}
	remaining := b.limit - int64(b.buf.Len())
	if remaining <= 0 {
		return 0, errWriterLimit
	}
	if int64(len(p)) > remaining {
		_, _ = b.buf.Write(p[:remaining])
		return int(remaining), errWriterLimit
	}
	return b.buf.Write(p)
}

func (b *hardLimitBuffer) Bytes() []byte { return b.buf.Bytes() }

type clippedBuffer struct {
	buf     bytes.Buffer
	limit   int64
	clipped bool
}

func (b *clippedBuffer) Write(p []byte) (int, error) {
	if b.limit <= 0 {
		return len(p), nil
	}
	remaining := b.limit - int64(b.buf.Len())
	if remaining <= 0 {
		b.clipped = true
		return len(p), nil
	}
	if int64(len(p)) > remaining {
		_, _ = b.buf.Write(p[:remaining])
		b.clipped = true
		return len(p), nil
	}
	return b.buf.Write(p)
}
