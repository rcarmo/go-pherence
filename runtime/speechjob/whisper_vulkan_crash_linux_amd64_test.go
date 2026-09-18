//go:build linux && amd64

package speechjob

import (
	"bufio"
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/models/whisper"
)

func TestVulkanJobQuarantineChild(t *testing.T) {
	mode := os.Getenv("SPEECHJOB_VULKAN_CHILD")
	if mode == "" {
		return
	}
	store, e := Open(os.Getenv("SPEECHJOB_VULKAN_DIR"), limits())
	if e != nil {
		t.Fatal(e)
	}
	var starts []int64
	infer := func(ctx context.Context, r whisper.SampleReader, total, first int64, emit func(whisper.WindowTranscript) error) error {
		e := fixtureWindows(&starts, 1)(ctx, r, total, first, emit)
		switch mode {
		case "infer-panic":
			panic("fixture")
		case "device-lost":
			return errors.Join(e, vk.ErrVulkanDeviceLost)
		case "uncertain":
			return errors.Join(e, vk.ErrVulkanUncertain)
		default:
			return errors.Join(e, vk.ErrVulkanInFlight)
		}
	}
	drain := func(context.Context, time.Duration) error {
		switch mode {
		case "drain-panic":
			panic("fixture")
		case "drain-error":
			return io.ErrClosedPipe
		case "drain-device":
			return vk.ErrVulkanDeviceLost
		case "drain-uncertain":
			return vk.ErrVulkanUncertain
		default:
			return errors.Join(vk.ErrVulkanInFlight, context.DeadlineExceeded)
		}
	}
	owner := mockVulkanJob(time.Millisecond, drain, func() error { t.Error("closed quarantined encoder"); return nil }, infer)
	done := make(chan error, 1)
	go func() {
		_, e := store.Run(context.Background(), os.Getenv("SPEECHJOB_VULKAN_ID"), config, []Stage{fixturePCMStage(1001), owner.Stage()}, nil)
		done <- e
	}()
	waitOwner(t, owner, func(s VulkanWhisperStatus) bool {
		if mode == "pending" {
			return s.Draining
		}
		return s.Quarantined
	})
	if e = store.Close(); !errors.Is(e, ErrBusy) {
		t.Fatal("store freed", e)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if e = owner.Close(ctx); !errors.Is(e, context.DeadlineExceeded) {
		t.Fatal("owner did not retain", e)
	}
	select {
	case e := <-done:
		t.Fatal("quarantine returned", e)
	default:
	}
	os.Stdout.WriteString("VULKAN_JOB_HELD\n")
	<-done
	t.Fatal("held job escaped")
}
func TestVulkanJobProcessKillRetainedWorkRecovery(t *testing.T) {
	for _, mode := range []string{"pending", "device-lost", "uncertain", "infer-panic", "drain-panic", "drain-error", "drain-device", "drain-uncertain"} {
		t.Run(mode, func(t *testing.T) {
			s, dir := openTest(t)
			job := createTest(t, s)
			s.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			child := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestVulkanJobQuarantineChild$", "-test.timeout=15s")
			child.Env = append(os.Environ(), "SPEECHJOB_VULKAN_CHILD="+mode, "SPEECHJOB_VULKAN_DIR="+dir, "SPEECHJOB_VULKAN_ID="+job.ID)
			stdout, e := child.StdoutPipe()
			if e != nil {
				t.Fatal(e)
			}
			var stderr strings.Builder
			child.Stderr = &stderr
			if e = child.Start(); e != nil {
				t.Fatal(e)
			}
			defer child.Process.Kill()
			scan := bufio.NewScanner(stdout)
			ready := false
			for scan.Scan() {
				if scan.Text() == "VULKAN_JOB_HELD" {
					ready = true
					break
				}
			}
			if !ready {
				child.Process.Kill()
				child.Wait()
				t.Fatal("barrier missing", stderr.String())
			}
			child.Process.Kill()
			if e = child.Wait(); e == nil {
				t.Fatal("expected process death")
			}
			s, e = Open(dir, limits())
			if e != nil {
				t.Fatal(e)
			}
			defer s.Close()
			stored, e := s.Get(job.ID)
			if e != nil || stored.Status != Failed || len(stored.Checkpoints) != 1 {
				t.Fatal(stored, e)
			}
			// Process teardown ended the MOCK native owner. Restart requires an explicit
			// run, and the durable window prefix survives; no failed-window ack exists.
			var starts []int64
			owner := mockVulkanJob(time.Millisecond, func(context.Context, time.Duration) error { return nil }, func() error { return nil }, fixtureWindows(&starts, -1))
			stored, e = s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(1001), owner.Stage()}, nil)
			if e != nil || stored.Status != Complete || len(starts) != 1 || starts[0] != 1 {
				t.Fatal(stored, e, starts)
			}
			owner.Close(context.Background())
		})
	}
}
