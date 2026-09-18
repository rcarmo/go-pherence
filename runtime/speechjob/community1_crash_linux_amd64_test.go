//go:build linux && amd64

package speechjob

import (
	"bufio"
	"context"
	c1 "github.com/rcarmo/go-pherence/models/speaker/community1"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func communityCrashStages(calls *int) []Stage {
	cfg := communityConfig()
	st := community1Stage(cfg, func(_ context.Context, _ c1.DiarizationPCMReader, total int64) (*c1.DiarizationPCMResult, error) {
		*calls++
		return communityFixtureResult(total, cfg), nil
	})
	return []Stage{fixturePCMStage(3361), textStage("asr-windows", "preserved"), st}
}
func TestCommunityJournalCrashChild(t *testing.T) {
	if os.Getenv("SPEECHJOB_COMMUNITY_CRASH") != "1" {
		return
	}
	s, e := Open(os.Getenv("SPEECHJOB_CRASH_DIR"), limits())
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	count := 0
	s.fault = func(p string) error {
		if p == os.Getenv("SPEECHJOB_CRASH_POINT") {
			count++
			target := 1
			if p == "payload-published" {
				target = 3
			}
			// running manifest, then active + checkpoint per three stages => 7
			if p == "manifest-renamed" {
				target = 7
			}
			if count == target {
				os.Stdout.WriteString("CRASH_READY\n")
				time.Sleep(time.Hour)
			}
		}
		return nil
	}
	calls := 0
	_, e = s.Run(context.Background(), os.Getenv("SPEECHJOB_CRASH_ID"), config, communityCrashStages(&calls), nil)
	t.Fatal("crash barrier not reached", e)
}
func TestCommunityProcessKillPublication(t *testing.T) {
	for _, point := range []string{"diarization-output-ready", "payload-published", "manifest-renamed"} {
		t.Run(point, func(t *testing.T) {
			s, dir := openTest(t)
			job := createTest(t, s)
			if e := s.Close(); e != nil {
				t.Fatal(e)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestCommunityJournalCrashChild$", "-test.timeout=30s")
			cmd.Env = append(os.Environ(), "SPEECHJOB_CRASH_DIR="+dir, "SPEECHJOB_CRASH_ID="+job.ID, "SPEECHJOB_CRASH_POINT="+point, "SPEECHJOB_COMMUNITY_CRASH=1")
			stdout, e := cmd.StdoutPipe()
			if e != nil {
				t.Fatal(e)
			}
			var stderr strings.Builder
			cmd.Stderr = &stderr
			if e = cmd.Start(); e != nil {
				t.Fatal(e)
			}
			scan := bufio.NewScanner(stdout)
			ready := false
			for scan.Scan() {
				if scan.Text() == "CRASH_READY" {
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
			wantCheckpoints := 2
			wantCalls := 1
			if point == "manifest-renamed" {
				wantCheckpoints = 3
				wantCalls = 0
			}
			if e != nil || job.Status != Failed || len(job.Checkpoints) != wantCheckpoints {
				t.Fatal(job, e)
			}
			r, e := s.OpenCheckpoint(context.Background(), job.ID, "asr-windows")
			if readAll(t, r, e) != "preserved" {
				t.Fatal("ASR lost")
			}
			calls := 0
			job, e = s.Run(context.Background(), job.ID, config, communityCrashStages(&calls), nil)
			if e != nil || job.Status != Complete || calls != wantCalls {
				t.Fatal(job, calls, e)
			}
			r, e = s.OpenCheckpoint(context.Background(), job.ID, "diarization")
			raw := readAll(t, r, e)
			if _, e = ReadDiarizationJSON(context.Background(), strings.NewReader(raw)); e != nil {
				t.Fatal(e)
			}
		})
	}
}
