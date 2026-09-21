//go:build linux && amd64

package speechjob

import (
	"bufio"
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	c1 "github.com/rcarmo/go-pherence/model/speaker/community1"
)

func TestVulkanCommunityQuarantineChild(t *testing.T) {
	mode := os.Getenv("SPEECHJOB_VULKAN_COMMUNITY_CHILD")
	if mode == "" {
		return
	}
	store, err := Open(os.Getenv("SPEECHJOB_VULKAN_COMMUNITY_DIR"), limits())
	if err != nil {
		t.Fatal(err)
	}
	cfg := communityConfig()
	cfg.ExecutionBackendSHA256 = hash([]byte("fixture Vulkan backend"))
	run := func(context.Context, c1.DiarizationPCMReader, int64) (*c1.DiarizationPCMResult, error) {
		switch mode {
		case "infer-panic":
			panic("fixture")
		case "device-lost":
			return nil, vk.ErrVulkanDeviceLost
		case "uncertain":
			return nil, vk.ErrVulkanUncertain
		default:
			return nil, vk.ErrVulkanInFlight
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
	s := newVulkanCommunity1Owner(time.Millisecond, run, drain, func() error { t.Error("closed quarantined model"); return nil })
	owner := &VulkanCommunity1Stage{s: s, stage: community1Stage(cfg, s.infer)}
	done := make(chan error, 1)
	go func() {
		_, err := store.Run(context.Background(), os.Getenv("SPEECHJOB_VULKAN_COMMUNITY_ID"), config, []Stage{fixturePCMStage(800), owner.Stage()}, nil)
		done <- err
	}()
	deadline := time.Now().Add(time.Second)
	for {
		status := owner.Status()
		if mode == "pending" && status.Draining || mode != "pending" && status.Quarantined {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal(status)
		}
		time.Sleep(time.Millisecond)
	}
	if err = store.Close(); !errors.Is(err, ErrBusy) {
		t.Fatal("store freed", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if err = owner.Close(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal("owner did not retain", err)
	}
	select {
	case err := <-done:
		t.Fatal("quarantine returned", err)
	default:
	}
	os.Stdout.WriteString("VULKAN_COMMUNITY_HELD\n")
	<-done
	t.Fatal("held job escaped")
}

func TestVulkanCommunityProcessKillRetainedWorkRecovery(t *testing.T) {
	for _, mode := range []string{"pending", "device-lost", "uncertain", "infer-panic", "drain-panic", "drain-error", "drain-device", "drain-uncertain"} {
		t.Run(mode, func(t *testing.T) {
			cfg := communityConfig()
			cfg.ExecutionBackendSHA256 = hash([]byte("fixture Vulkan backend"))
			store, dir := openTest(t)
			job := createTest(t, store)
			store.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			child := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestVulkanCommunityQuarantineChild$", "-test.timeout=15s")
			child.Env = append(os.Environ(), "SPEECHJOB_VULKAN_COMMUNITY_CHILD="+mode, "SPEECHJOB_VULKAN_COMMUNITY_DIR="+dir, "SPEECHJOB_VULKAN_COMMUNITY_ID="+job.ID)
			stdout, err := child.StdoutPipe()
			if err != nil {
				t.Fatal(err)
			}
			var stderr strings.Builder
			child.Stderr = &stderr
			if err = child.Start(); err != nil {
				t.Fatal(err)
			}
			defer child.Process.Kill()
			scan := bufio.NewScanner(stdout)
			ready := false
			for scan.Scan() {
				if scan.Text() == "VULKAN_COMMUNITY_HELD" {
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
			if err = child.Wait(); err == nil {
				t.Fatal("expected process death")
			}
			store, err = Open(dir, limits())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			stored, err := store.Get(job.ID)
			if err != nil || stored.Status != Failed || len(stored.Checkpoints) != 1 {
				t.Fatal(stored, err)
			}
			// Process teardown ended the mock native owner. Explicit retry uses the
			// same backend-bound stage identity and restarts whole-result diarization.
			s := newVulkanCommunity1Owner(time.Millisecond, func(context.Context, c1.DiarizationPCMReader, int64) (*c1.DiarizationPCMResult, error) {
				return communityFixtureResult(800, cfg), nil
			}, func(context.Context, time.Duration) error { return nil }, func() error { return nil })
			owner := &VulkanCommunity1Stage{s: s, stage: community1Stage(cfg, s.infer)}
			stored, err = store.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(800), owner.Stage()}, nil)
			if err != nil || stored.Status != Complete || len(stored.Checkpoints) != 2 {
				t.Fatal(stored, err)
			}
			owner.Close(context.Background())
			if _, err := os.Stat(filepath.Join(dir, job.ID)); err != nil {
				t.Fatal(err)
			}
		})
	}
}
