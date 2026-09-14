//go:build linux || darwin || freebsd || openbsd || netbsd || dragonfly

package speechjob

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func crashQueue(t *testing.T, s *Store, dir string, calls *int) *Queue {
	t.Helper()
	return openQueueTest(t, s, dir, SerialAdmission(), func(Manifest) ([]Stage, error) {
		return []Stage{stage("transcript", func(_ context.Context, _ *Input, w io.Writer) error {
			*calls++
			_, e := io.WriteString(w, "saved")
			return e
		})}, nil
	})
}
func TestQueueCrashChild(t *testing.T) {
	dir := os.Getenv("SPEECHQUEUE_CRASH_DIR")
	if dir == "" {
		return
	}
	s, e := Open(filepath.Join(dir, "store"), limits())
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	calls := 0
	q := crashQueue(t, s, filepath.Join(dir, "queue"), &calls)
	point := os.Getenv("SPEECHQUEUE_CRASH_POINT")
	count := 0
	q.fault = func(p string) error {
		if p == point {
			count++
			target := 1
			if point == "queue-renamed" {
				target = 3
			}
			if count == target {
				fmt.Println("QUEUE_CRASH_READY")
				time.Sleep(time.Hour)
			}
		}
		return nil
	}
	// For queue-renamed: enqueue publish, running claim, terminal publish.
	if _, e = q.Enqueue(context.Background(), os.Getenv("SPEECHQUEUE_CRASH_ID"), false); e != nil {
		t.Fatal(e)
	}
	if point == "queue-pending" {
		fmt.Println("QUEUE_CRASH_READY")
		time.Sleep(time.Hour)
	}
	if e = q.Start(context.Background()); e != nil {
		t.Fatal(e)
	}
	<-q.done
	t.Fatal("barrier not reached")
}
func TestQueueProcessKillRecovery(t *testing.T) {
	for _, point := range []string{"queue-pending", "queue-claimed", "queue-job-committed", "queue-renamed"} {
		t.Run(point, func(t *testing.T) {
			dir := t.TempDir()
			s, e := Open(filepath.Join(dir, "store"), limits())
			if e != nil {
				t.Fatal(e)
			}
			job := createTest(t, s)
			s.Close()
			// Seed empty journal before fault points are installed in the child.
			s, e = Open(filepath.Join(dir, "store"), limits())
			if e != nil {
				t.Fatal(e)
			}
			n := 0
			q := crashQueue(t, s, filepath.Join(dir, "queue"), &n)
			q.Close()
			s.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestQueueCrashChild$", "-test.timeout=30s")
			cmd.Env = append(os.Environ(), "SPEECHQUEUE_CRASH_DIR="+dir, "SPEECHQUEUE_CRASH_ID="+job.ID, "SPEECHQUEUE_CRASH_POINT="+point)
			pipe, e := cmd.StdoutPipe()
			if e != nil {
				t.Fatal(e)
			}
			var stderr strings.Builder
			cmd.Stderr = &stderr
			if e = cmd.Start(); e != nil {
				t.Fatal(e)
			}
			scan := bufio.NewScanner(pipe)
			ready := false
			for scan.Scan() {
				if scan.Text() == "QUEUE_CRASH_READY" {
					ready = true
					break
				}
			}
			if !ready {
				cmd.Process.Kill()
				cmd.Wait()
				t.Fatal("missing barrier", stderr.String())
			}
			cmd.Process.Kill()
			if e = cmd.Wait(); e == nil {
				t.Fatal("expected kill")
			}
			s, e = Open(filepath.Join(dir, "store"), limits())
			if e != nil {
				t.Fatal(e)
			}
			defer s.Close()
			calls := 0
			q = crashQueue(t, s, filepath.Join(dir, "queue"), &calls)
			entries, e := q.List()
			if e != nil || len(entries) != 1 {
				t.Fatal(entries, e)
			}
			expected := QueuePending
			if point == "queue-claimed" || point == "queue-job-committed" {
				expected = QueueInterrupted
			}
			if point == "queue-renamed" {
				expected = QueueSucceeded
			}
			if entries[0].Status != expected {
				t.Fatal(entries)
			}
			if point == "queue-job-committed" {
				// Job commit preceded queue terminal publication. Never replay it on
				// restart or manual retry; inspect the complete store job then forget.
				if _, e = q.Enqueue(context.Background(), job.ID, true); !errors.Is(e, ErrQueueState) {
					t.Fatal("committed job replay", e)
				}
				q.Start(context.Background())
				q.Shutdown(context.Background())
				if calls != 0 {
					t.Fatal(calls)
				}
				return
			}
			if expected == QueueInterrupted {
				if _, e = q.Enqueue(context.Background(), job.ID, false); !errors.Is(e, ErrQueueState) {
					t.Fatal(e)
				}
				if _, e = q.Enqueue(context.Background(), job.ID, true); e != nil {
					t.Fatal(e)
				}
			}
			q.Start(context.Background())
			if expected != QueueSucceeded {
				queueWait(t, q, job.ID, QueueSucceeded)
			}
			q.Shutdown(context.Background())
			want := 1
			if expected == QueueSucceeded {
				want = 0
			}
			if calls != want {
				t.Fatal(calls, want)
			}
		})
	}
}
