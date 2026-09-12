//go:build linux && amd64

package speechjob

import (
	"bufio"
	"context"
	"io"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func TestReconcileCrashChild(t *testing.T) {
	if os.Getenv("SPEECHJOB_RECONCILE_CRASH") != "1" {
		return
	}
	s, e := Open(os.Getenv("SPEECHJOB_CRASH_DIR"), limits())
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	s.fault = func(p string) error {
		if p == os.Getenv("SPEECHJOB_CRASH_POINT") {
			os.Stdout.WriteString("CRASH_READY\n")
			time.Sleep(time.Hour)
		}
		return nil
	}
	_, e = s.Run(context.Background(), os.Getenv("SPEECHJOB_CRASH_ID"), config, speakerPipeline(t, false), nil)
	t.Fatal("barrier not reached", e)
}
func TestReconcileProcessKillOutputs(t *testing.T) {
	for _, tc := range []struct {
		point       string
		checkpoints int
	}{{"transcript-output-ready", 2}, {"speaker-transcript-output-ready", 5}, {"speaker-vtt-output-ready", 6}} {
		t.Run(tc.point, func(t *testing.T) {
			s, dir := openTest(t)
			job := createTest(t, s)
			if e := s.Close(); e != nil {
				t.Fatal(e)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestReconcileCrashChild$", "-test.timeout=30s")
			cmd.Env = append(os.Environ(), "SPEECHJOB_CRASH_DIR="+dir, "SPEECHJOB_CRASH_ID="+job.ID, "SPEECHJOB_CRASH_POINT="+tc.point, "SPEECHJOB_RECONCILE_CRASH=1")
			stdout, e := cmd.StdoutPipe()
			if e != nil {
				t.Fatal(e)
			}
			var stderr strings.Builder
			cmd.Stderr = &stderr
			if e = cmd.Start(); e != nil {
				t.Fatal(e)
			}
			scanner := bufio.NewScanner(stdout)
			ready := false
			for scanner.Scan() {
				if scanner.Text() == "CRASH_READY" {
					ready = true
					break
				}
			}
			if !ready {
				cmd.Process.Kill()
				cmd.Wait()
				t.Fatal("barrier", stderr.String())
			}
			if e = cmd.Process.Kill(); e != nil {
				t.Fatal(e)
			}
			if e = cmd.Wait(); e == nil {
				t.Fatal("expected kill")
			}
			s, e = Open(dir, limits())
			if e != nil {
				t.Fatal(e)
			}
			defer s.Close()
			job, e = s.Get(job.ID)
			if e != nil || job.Status != Failed || len(job.Checkpoints) != tc.checkpoints {
				t.Fatal(job, e)
			}
			stages := speakerPipeline(t, false)
			for i := 0; i < tc.checkpoints; i++ {
				stages[i].Run = func(context.Context, *Input, io.Writer) error { t.Fatal("repeated committed work"); return nil }
			}
			job, e = s.Run(context.Background(), job.ID, config, stages, nil)
			if e != nil || job.Status != Complete {
				t.Fatal(job, e)
			}
			r, e := s.OpenCheckpoint(context.Background(), job.ID, "speaker-vtt")
			if !strings.Contains(readAll(t, r, e), "NOTE Experimental") {
				t.Fatal("missing warning")
			}
		})
	}
}
