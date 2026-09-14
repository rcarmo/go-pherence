//go:build linux && amd64

package speechjob

import (
	"bufio"
	"context"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func TestWhisperJournalCrashChild(t *testing.T) {
	if os.Getenv("SPEECHJOB_WHISPER_JOURNAL") != "1" {
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
	var starts []int64
	_, e = s.Run(context.Background(), os.Getenv("SPEECHJOB_CRASH_ID"), config, []Stage{fixturePCMStage(1001), fixtureWhisperStage(&starts, -1)}, nil)
	t.Fatal("crash barrier not reached", e)
}

func TestWhisperProcessKillWindowJournal(t *testing.T) {
	for _, point := range []string{"window-payload-published", "window-ack-published"} {
		t.Run(point, func(t *testing.T) {
			s, dir := openTest(t)
			job := createTest(t, s)
			if e := s.Close(); e != nil {
				t.Fatal(e)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestWhisperJournalCrashChild$", "-test.timeout=30s")
			cmd.Env = append(os.Environ(), "SPEECHJOB_CRASH_DIR="+dir, "SPEECHJOB_CRASH_ID="+job.ID, "SPEECHJOB_CRASH_POINT="+point, "SPEECHJOB_WHISPER_JOURNAL=1")
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
				t.Fatal("barrier missing", stderr.String())
			}
			if e = cmd.Process.Kill(); e != nil {
				t.Fatal(e)
			}
			if e = cmd.Wait(); e == nil {
				t.Fatal("expected SIGKILL")
			}
			s, e = Open(dir, limits())
			if e != nil {
				t.Fatal(e)
			}
			defer s.Close()
			job, e = s.Get(job.ID)
			if e != nil || job.Status != Failed || len(job.Checkpoints) != 1 {
				t.Fatal(job, e)
			}
			var starts []int64
			job, e = s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(1001), fixtureWhisperStage(&starts, -1)}, nil)
			want := int64(0)
			if point == "window-ack-published" {
				want = 1
			}
			if e != nil || job.Status != Complete || len(starts) != 1 || starts[0] != want {
				t.Fatal(job, starts, e)
			}
			r, e := s.OpenCheckpoint(context.Background(), job.ID, "asr-windows")
			if len(strings.Split(strings.TrimSpace(readAll(t, r, e)), "\n")) != 6 {
				t.Fatal("lost windows")
			}
		})
	}
}
