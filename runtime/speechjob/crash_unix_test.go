//go:build linux || darwin || freebsd || openbsd || netbsd || dragonfly

package speechjob

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The child blocks at one actual publication boundary. Parent SIGKILL exercises
// process death without defers; it does not simulate sudden hardware power loss.
func TestStoreCrashChild(t *testing.T) {
	dir := os.Getenv("SPEECHJOB_CRASH_DIR")
	if dir == "" {
		return
	}
	s, e := Open(dir, limits())
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	point := os.Getenv("SPEECHJOB_CRASH_POINT")
	count := 0
	s.fault = func(p string) error {
		if p == point {
			count++
			target := 1
			if strings.HasPrefix(p, "manifest-") {
				target = 3
			}
			if count == target {
				os.Stdout.WriteString("CRASH_READY\n")
				time.Sleep(time.Hour)
			}
		}
		return nil
	}
	st := textStage("asr", "verified transcript")
	if raw := os.Getenv("SPEECHJOB_DECODE_CONFIG"); raw != "" {
		var cfg FFmpegDecodeConfig
		if e = json.Unmarshal([]byte(raw), &cfg); e != nil {
			t.Fatal(e)
		}
		st = newDecodeStage(cfg, &decodeFixtureAdapter{})
	}
	_, e = s.Run(context.Background(), os.Getenv("SPEECHJOB_CRASH_ID"), config, []Stage{st}, nil)
	if e != nil {
		t.Fatal(e)
	}
	t.Fatal("crash barrier not reached")
}
func TestStoreProcessKillRecovery(t *testing.T) {
	for _, point := range []string{"payload-synced", "payload-published", "manifest-synced", "manifest-renamed"} {
		t.Run(point, func(t *testing.T) {
			dir := filepath.Join(t.TempDir(), "private")
			s, e := Open(dir, limits())
			if e != nil {
				t.Fatal(e)
			}
			m := createTest(t, s)
			if e = s.Close(); e != nil {
				t.Fatal(e)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestStoreCrashChild$", "-test.timeout=30s")
			cmd.Env = append(os.Environ(), "SPEECHJOB_CRASH_DIR="+dir, "SPEECHJOB_CRASH_ID="+m.ID, "SPEECHJOB_CRASH_POINT="+point)
			stdout, e := cmd.StdoutPipe()
			if e != nil {
				t.Fatal(e)
			}
			var stderr strings.Builder
			cmd.Stderr = &stderr
			if e = cmd.Start(); e != nil {
				t.Fatal(e)
			}
			ready := false
			scanner := bufio.NewScanner(stdout)
			for scanner.Scan() {
				if scanner.Text() == "CRASH_READY" {
					ready = true
					break
				}
			}
			if !ready {
				cmd.Process.Kill()
				cmd.Wait()
				t.Fatal("child did not reach barrier", point, stderr.String(), ctx.Err())
			}
			// Separate process lock must still hold while its writer is alive.
			if other, e := Open(dir, limits()); !errors.Is(e, ErrBusy) {
				if other != nil {
					other.Close()
				}
				t.Fatal("cross-process lock", e)
			}
			if e = cmd.Process.Kill(); e != nil {
				t.Fatal(e)
			}
			if e = cmd.Wait(); e == nil {
				t.Fatal("kill reported success")
			}
			s, e = Open(dir, limits())
			if e != nil {
				t.Fatal(e)
			}
			defer s.Close()
			m, e = s.Get(m.ID)
			if e != nil || m.Status != Failed || !strings.Contains(m.Error, "interrupted") {
				t.Fatal("recovery", point, m, e)
			}
			calls := 0
			st := stage("asr", func(_ context.Context, _ *Input, w io.Writer) error {
				calls++
				_, e := io.WriteString(w, "verified transcript")
				return e
			})
			m, e = s.Run(context.Background(), m.ID, config, []Stage{st}, nil)
			if e != nil || m.Status != Complete {
				t.Fatal(point, e)
			}
			expected := 1
			if point == "manifest-renamed" {
				expected = 0
			}
			if calls != expected {
				t.Fatal("wrong resume point", point, calls)
			}
			r, e := s.OpenCheckpoint(context.Background(), m.ID, "asr")
			if readAll(t, r, e) != "verified transcript" {
				t.Fatal("lost payload")
			}
			used, _, e := s.Usage()
			if e != nil || used <= m.Input.Bytes {
				t.Fatal("quota ignores retained orphan", e)
			}
		})
	}
}

func TestDecodeProcessKillScratchRetention(t *testing.T) {
	for _, point := range []string{"decode-input-ready", "decode-output-ready"} {
		t.Run(point, func(t *testing.T) {
			s, dir := openTest(t)
			m := createTest(t, s)
			cfg := decodeConfig(t)
			cfgJSON, e := json.Marshal(cfg)
			if e != nil {
				t.Fatal(e)
			}
			s.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestStoreCrashChild$", "-test.timeout=30s")
			cmd.Env = append(os.Environ(), "SPEECHJOB_CRASH_DIR="+dir, "SPEECHJOB_CRASH_ID="+m.ID, "SPEECHJOB_CRASH_POINT="+point, "SPEECHJOB_DECODE_CONFIG="+string(cfgJSON))
			stdout, e := cmd.StdoutPipe()
			if e != nil {
				t.Fatal(e)
			}
			var stderr strings.Builder
			cmd.Stderr = &stderr
			if e = cmd.Start(); e != nil {
				t.Fatal(e)
			}
			ready := false
			scanner := bufio.NewScanner(stdout)
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
			m, e = s.Get(m.ID)
			if e != nil || m.Status != Failed || len(m.Checkpoints) != 0 {
				t.Fatal(m, e)
			}
			retained := ""
			entries, _ := os.ReadDir(filepath.Join(dir, m.ID))
			for _, entry := range entries {
				if strings.HasPrefix(entry.Name(), ".work-decode-") {
					retained = entry.Name()
					st, e := os.Stat(filepath.Join(dir, m.ID, entry.Name(), "input.wav"))
					if e != nil || st.Size() != m.Input.Bytes {
						t.Fatal("scratch source lost", e)
					}
				}
			}
			if retained == "" {
				t.Fatal("killed stage scratch not retained")
			}
			inventory, e := s.Inventory()
			if e != nil || len(inventory) != 1 || inventory[0].Bytes <= 2*m.Input.Bytes {
				t.Fatal(inventory, e)
			}
			m, e = s.Run(context.Background(), m.ID, config, []Stage{newDecodeStage(cfg, &decodeFixtureAdapter{})}, nil)
			if e != nil || m.Status != Complete {
				t.Fatal(m, e)
			}
			// Retry does not silently delete retained old scratch. Explicit job deletion does.
			if _, e := os.Stat(filepath.Join(dir, m.ID, retained, "input.wav")); e != nil {
				t.Fatal("retry removed retained scratch", e)
			}
			if e = s.Delete(context.Background(), m.ID); e != nil {
				t.Fatal(e)
			}
			if _, e = os.Stat(filepath.Join(dir, m.ID)); !errors.Is(e, os.ErrNotExist) {
				t.Fatal("delete", e)
			}
		})
	}
}
