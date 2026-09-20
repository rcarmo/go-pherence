// Package commandcapture runs owned helper commands with bounded captured output.
package commandcapture

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"sync"
	"time"
)

var ErrOutputLimit = errors.New("command output limit exceeded")

// Run captures stdout and stderr under one aggregate limit. env=nil inherits the
// caller environment. Cancellation/overflow kills the owned process group on
// Linux and waits for command/pipes before returning. Other OSes kill the direct
// child only. This does not sandbox the executable or bound its CPU/RSS/files.
func Run(ctx context.Context, path string, args, env []string, limit int) ([]byte, []byte, error) {
	if ctx == nil || path == "" || limit <= 0 {
		return nil, nil, fmt.Errorf("invalid command capture arguments")
	}
	owned, cancel := context.WithCancel(ctx)
	defer cancel()
	cmd := exec.CommandContext(owned, path, args...)
	cmd.Env = env
	cmd.WaitDelay = 2 * time.Second
	configureOwnedCommand(cmd)
	budget := &captureBudget{remaining: limit, cancel: cancel}
	stdout, stderr := &captureWriter{budget: budget}, &captureWriter{budget: budget}
	cmd.Stdout, cmd.Stderr = stdout, stderr
	err := cmd.Run()
	if ctx.Err() != nil {
		err = ctx.Err()
	} else if budget.exceeded {
		err = ErrOutputLimit
	}
	return stdout.buf.Bytes(), stderr.buf.Bytes(), err
}

type captureBudget struct {
	mu        sync.Mutex
	remaining int
	exceeded  bool
	cancel    context.CancelFunc
}
type captureWriter struct {
	budget *captureBudget
	buf    bytes.Buffer
}

func (w *captureWriter) Write(p []byte) (int, error) {
	b := w.budget
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(p) > b.remaining {
		n := b.remaining
		if n > 0 {
			_, _ = w.buf.Write(p[:n])
			b.remaining = 0
		}
		b.exceeded = true
		b.cancel()
		return n, ErrOutputLimit
	}
	n, err := w.buf.Write(p)
	b.remaining -= n
	return n, err
}
