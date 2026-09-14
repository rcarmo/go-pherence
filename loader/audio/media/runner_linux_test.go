//go:build linux

package media

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestExecRunnerCancellationKillsOwnedProcessGroup(t *testing.T) {
	dir := t.TempDir()
	pidFile := filepath.Join(dir, "child.pid")
	markerFile := filepath.Join(dir, "descendant-alive")
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- (execRunner{}).Run(ctx, Command{Path: "/bin/sh", Args: []string{"-c", "(sleep 1; echo alive > \"$2\") & echo $! > \"$1\"; wait", "sh", pidFile, markerFile}})
	}()
	deadline := time.Now().Add(2 * time.Second)
	for {
		if _, err := os.Stat(pidFile); err == nil {
			break
		}
		if time.Now().After(deadline) {
			cancel()
			t.Fatal("child did not start")
		}
		time.Sleep(10 * time.Millisecond)
	}
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel error=%v", err)
	}
	data, err := os.ReadFile(pidFile)
	if err != nil {
		t.Fatal(err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		t.Fatal(err)
	}
	// A killed descendant may briefly remain as a reparented zombie, so kill(0)
	// alone is not a portable liveness assertion. It must never run delayed work.
	time.Sleep(1200 * time.Millisecond)
	if _, err := os.Stat(markerFile); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("descendant %d survived cancellation and wrote marker: %v", pid, err)
	}
	if err := syscall.Kill(pid, 0); err == nil {
		stat, readErr := os.ReadFile(filepath.Join("/proc", strconv.Itoa(pid), "stat"))
		if readErr == nil && !strings.Contains(string(stat), ") Z ") {
			t.Fatalf("descendant %d remains live: %s", pid, stat)
		}
	}
}

func TestExecRunnerOwnProcessGroup(t *testing.T) {
	var out strings.Builder
	if err := (execRunner{}).Run(context.Background(), Command{Path: "/bin/sh", Args: []string{"-c", "printf '%s %s' $$ $(ps -o pgid= -p $$)"}, Stdout: &out}); err != nil {
		t.Fatal(err)
	}
	fields := strings.Fields(out.String())
	if len(fields) != 2 || fields[0] != fields[1] {
		t.Fatalf("pid/pgid=%q", out.String())
	}
}
