package commandcapture

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

func TestCaptureHelper(t *testing.T) {
	mode := os.Getenv("GO_PHERENCE_CAPTURE_HELPER")
	if mode == "" {
		return
	}
	switch mode {
	case "normal":
		fmt.Fprint(os.Stdout, "hello")
		fmt.Fprint(os.Stderr, "err")
	case "large":
		for i := 0; i < 128; i++ {
			fmt.Fprint(os.Stdout, strings.Repeat("x", 8192))
			fmt.Fprint(os.Stderr, strings.Repeat("y", 8192))
		}
	case "sleep":
		time.Sleep(30 * time.Second)
	case "fail":
		fmt.Fprint(os.Stderr, "failure")
		os.Exit(3)
	}
	os.Exit(0)
}
func helper(t *testing.T, mode string, limit int, ctx context.Context) ([]byte, []byte, error) {
	t.Helper()
	return Run(ctx, os.Args[0], []string{"-test.run=^TestCaptureHelper$"}, append(os.Environ(), "GO_PHERENCE_CAPTURE_HELPER="+mode), limit)
}
func TestCaptureBoundaries(t *testing.T) {
	out, stderr, err := helper(t, "normal", 8, context.Background())
	if err != nil || string(out) != "hello" || string(stderr) != "err" {
		t.Fatal(string(out), string(stderr), err)
	}
	out, stderr, err = helper(t, "large", 4096, context.Background())
	if !errors.Is(err, ErrOutputLimit) || len(out)+len(stderr) > 4096 {
		t.Fatal(len(out), len(stderr), err)
	}
	out, stderr, err = helper(t, "fail", 64, context.Background())
	if err == nil || string(stderr) != "failure" {
		t.Fatal(string(out), string(stderr), err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, _, err = helper(t, "sleep", 64, ctx)
	if !errors.Is(err, context.DeadlineExceeded) || time.Since(start) > 4*time.Second {
		t.Fatal(err, time.Since(start))
	}
	if _, _, err = Run(nil, "anything", nil, nil, 1); err == nil {
		t.Fatal("nil context accepted")
	}
}
