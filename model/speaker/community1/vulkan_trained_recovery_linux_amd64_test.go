//go:build linux && amd64

package community1

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
)

type trainedRecoveryReader struct {
	DiarizationPCMReader
	onSecondRead func()
	mu           sync.Mutex
	reads        int
}

func (r *trainedRecoveryReader) ReadSamplesAt(ctx context.Context, dst []float32, start int64) (int, error) {
	r.mu.Lock()
	r.reads++
	reads := r.reads
	r.mu.Unlock()
	if reads == 2 && r.onSecondRead != nil {
		r.onSecondRead()
	}
	return r.DiarizationPCMReader.ReadSamplesAt(ctx, dst, start)
}

func trainedVulkanRecoveryChild(t *testing.T, mode string) {
	t.Helper()
	if !vk.VulkanInit() {
		t.Fatal("native Vulkan initialisation failed")
	}
	device := vk.VulkanDeviceName()
	want := os.Getenv("GO_PHERENCE_VULKAN_DEVICE")
	lower := strings.ToLower(device)
	if want == "" || !strings.Contains(device, want) || strings.Contains(lower, "llvmpipe") || strings.Contains(lower, "lavapipe") {
		t.Fatal("unexpected physical device", device)
	}
	ctx := context.Background()
	segmentation, filters, embedding, prepared := loadTrainedDiarizationModels(t, ctx)
	pcm := openTrainedDiarizationPCM(t, ctx)
	before := vk.VulkanMemoryStats()
	model, err := NewVulkanDiarization(ctx, segmentation, filters, embedding, prepared.Model, 160000)
	if err != nil {
		t.Fatal(err)
	}
	reader := &trainedRecoveryReader{DiarizationPCMReader: pcm}
	if mode == "kill-after-window" {
		reader.onSecondRead = func() {
			fmt.Println("TRAINED_VULKAN_WINDOW_COMPLETE")
			time.Sleep(time.Hour)
		}
	}
	cfg := DiarizationPCMConfig{WindowSamples: 160000, StepSamples: 16000, MinimumEmbeddingSamples: 400, ExcludeOverlap: true, MinSpeakers: 1, MaxSpeakers: 64, AHCThreshold: .6, Fa: .07, Fb: .8, Constrained: true, TiePolicy: LowestIndexTies}
	result, err := model.RunPCM(ctx, reader, 480000, cfg, SincNetSIMDFMA, HeadSIMD)
	if mode == "kill-after-window" {
		t.Fatal("kill barrier escaped", result, err)
	}
	if err != nil {
		nativeCommunityClose(t, model)
		t.Fatal(err)
	}
	if len(result.Windows) != 21 || result.Postprocess.Path != "clustered" || result.Postprocess.TrainingRows != 37 || result.Postprocess.Clusters != 2 || len(result.Postprocess.FullTurns) != 13 || len(result.Postprocess.ExclusiveTurns) != 12 || len(result.Postprocess.Timeline.AmbiguousFrames) != 84 {
		nativeCommunityClose(t, model)
		t.Fatal("recovery result contract")
	}
	nativeCommunityClose(t, model)
	after := vk.VulkanMemoryStats()
	if after.Bytes != before.Bytes || after.Allocations != before.Allocations || after.InFlight || after.Uncertain || after.DeviceLost {
		t.Fatalf("recovery allocation/state leak: before%+v after%+v", before, after)
	}
	fmt.Printf("TRAINED_VULKAN_RECOVERED device=%q reads=%d\n", device, reader.reads)
}

func TestVulkanCommunity1TrainedRecoveryChild(t *testing.T) {
	mode := os.Getenv("GO_PHERENCE_VULKAN_COMMUNITY_RECOVERY_CHILD")
	if mode == "" {
		return
	}
	if mode != "kill-after-window" && mode != "recover" {
		t.Fatal("invalid recovery child mode")
	}
	trainedVulkanRecoveryChild(t, mode)
}

func TestVulkanCommunity1TrainedProcessRecovery(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_COMMUNITY_RECOVERY") != "1" {
		t.Skip("set GO_PHERENCE_TEST_VULKAN_COMMUNITY_RECOVERY=1 in an authorised compute window")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 2*time.Minute {
		t.Fatal("trained Vulkan process recovery requires go test -timeout of at most 2m")
	}
	run := func(mode string, waitFor string) (string, error) {
		ctx, cancel := context.WithTimeout(context.Background(), 75*time.Second)
		defer cancel()
		child := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestVulkanCommunity1TrainedRecoveryChild$", "-test.timeout=70s", "-test.v")
		child.Env = append(os.Environ(), "GO_PHERENCE_VULKAN_COMMUNITY_RECOVERY_CHILD="+mode)
		stdout, err := child.StdoutPipe()
		if err != nil {
			return "", err
		}
		var stderr strings.Builder
		child.Stderr = &stderr
		if err = child.Start(); err != nil {
			return "", err
		}
		scan := bufio.NewScanner(stdout)
		var output strings.Builder
		for scan.Scan() {
			line := scan.Text()
			output.WriteString(line)
			output.WriteByte('\n')
			if waitFor != "" && strings.Contains(line, waitFor) {
				if err = child.Process.Kill(); err != nil {
					return output.String(), err
				}
				err = child.Wait()
				if err == nil {
					return output.String(), errors.New("expected killed trained Vulkan child")
				}
				return output.String(), nil
			}
		}
		if err = scan.Err(); err != nil {
			_ = child.Process.Kill()
			_ = child.Wait()
			return output.String(), err
		}
		err = child.Wait()
		if waitFor != "" {
			return output.String(), fmt.Errorf("barrier missing: %w: %s", err, stderr.String())
		}
		if err != nil {
			return output.String(), fmt.Errorf("recovery child: %w: %s", err, stderr.String())
		}
		return output.String(), nil
	}
	killed, err := run("kill-after-window", "TRAINED_VULKAN_WINDOW_COMPLETE")
	if err != nil {
		t.Fatal(err, killed)
	}
	recovered, err := run("recover", "")
	if err != nil {
		t.Fatal(err, recovered)
	}
	if !strings.Contains(recovered, "TRAINED_VULKAN_RECOVERED") {
		t.Fatal("fresh process did not report recovery", recovered)
	}
	if strings.Contains(recovered, "FAIL") || !strings.Contains(recovered, "PASS") {
		t.Fatal("fresh process test failed", recovered)
	}
	t.Logf("TRAINED_VULKAN_PROCESS_RECOVERY killed_after_first_window=true fresh_process_complete=true")
}
