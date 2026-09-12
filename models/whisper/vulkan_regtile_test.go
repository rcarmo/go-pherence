package whisper

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/audio/media"
)

// Candidate whole-encoder/speech check. Public encoder defaults remain baseline.
func TestVulkanTurboRegTile(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_TURBO_REGTILE") != "1" {
		t.Skip("explicit Turbo candidate compute window required")
	}
	deadline, ok := t.Deadline()
	if !ok || time.Until(deadline) > 300*time.Second {
		t.Fatal("timeout<=300s")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 270*time.Second)
	defer cancel()
	model, tok, policy := pinnedTurboSpeechModel(t, ctx)
	if !vk.VulkanInit() {
		t.Fatal("Vulkan unavailable")
	}
	want := os.Getenv("GO_PHERENCE_VULKAN_DEVICE")
	if want == "" || !strings.Contains(vk.VulkanDeviceName(), want) {
		t.Fatal("unexpected device")
	}
	before := vk.VulkanMemoryStats()
	if before.Allocations != 0 {
		t.Fatal("isolatedprocessrequired")
	}
	if err := vk.VulkanSetMemoryBudget(vk.VulkanMemoryBudget{MaxBytes: 8 << 30, MaxAllocations: 80}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := vk.VulkanSetMemoryBudget(before.Budget); err != nil {
			t.Error(err)
		}
	})
	t.Cleanup(func() { nativeEncoderMemory(t, before) })
	baseline, err := NewVulkanEncoder(ctx, model.Encoder, 3000)
	if baseline != nil {
		t.Cleanup(func() {
			if err := baseline.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := newVulkanEncoderVariant(ctx, model.Encoder, 3000, vk.NewVkF32Plan, true)
	if candidate != nil {
		t.Cleanup(func() {
			if err := candidate.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Fatal(err)
	}
	ffprobe, err := exec.LookPath("ffprobe")
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := media.NewFFmpeg(media.Config{FFmpegPath: ffmpeg, FFprobePath: ffprobe})
	if err != nil {
		t.Fatal(err)
	}
	fixtures, err := turboPublicFixtures("", os.Getenv("GO_PHERENCE_MINDS_FIXTURE_DIR"), os.Getenv("GO_PHERENCE_WHISPER_JFK_PATH"))
	if err != nil {
		t.Fatal(err)
	}
	type pair struct {
		fixture publicSpeechFixture
		decoded media.DecodeResult
	}
	var pairs []pair
	for _, f := range fixtures {
		pinnedSpeechFile(t, f.File, f.SHA256)
		d, err := adapter.DecodeToFile(ctx, f.File, filepath.Join(t.TempDir(), f.Name+".wav"))
		if err != nil {
			t.Fatal(err)
		}
		if int64(d.Timeline.Samples) != f.Samples {
			t.Fatal("PCM extent")
		}
		pairs = append(pairs, pair{f, d})
	}
	timingIndex := -1
	for i, p := range pairs {
		if p.fixture.Name == "minds-pt-0" {
			timingIndex = i
			break
		}
	}
	if timingIndex < 0 {
		t.Fatal("missing explicit timing fixture minds-pt-0")
	}
	timingPair := pairs[timingIndex]
	reader, err := media.OpenCanonicalPCM(ctx, timingPair.decoded.Path)
	if err != nil {
		t.Fatal(err)
	}
	pcm := make([]float32, timingPair.fixture.Samples)
	n, readErr := reader.ReadSamplesAt(ctx, pcm, 0)
	closeErr := reader.Close()
	if n != len(pcm) || readErr != nil || closeErr != nil {
		t.Fatal(n, readErr, closeErr)
	}
	padded := make([]float32, 480000)
	copy(padded, pcm)
	mel, _, err := MelFlatFromSamplesCheckedContext(ctx, padded, model.Config)
	if err != nil {
		t.Fatal(err)
	}
	reference, err := baseline.Forward(ctx, mel)
	if err != nil {
		t.Fatal(err)
	}
	got, err := candidate.Forward(ctx, mel)
	if err != nil {
		t.Fatal(err)
	}
	equalContextFloats(t, reference, got)
	t.Log("REGTILE_ENCODER initial full trained Turbo output bit-exact")
	durations := map[string][]int64{"baseline": {}, "candidate": {}}
	for block := 0; block < 2; block++ {
		order := []string{"baseline", "candidate", "candidate", "baseline"}
		if block%2 == 1 {
			order = []string{"candidate", "baseline", "baseline", "candidate"}
		}
		for _, name := range order {
			e := baseline
			if name == "candidate" {
				e = candidate
			}
			start := time.Now()
			out, err := e.Forward(ctx, mel)
			ns := time.Since(start).Nanoseconds()
			if err != nil {
				t.Fatal(err)
			}
			equalContextFloats(t, reference, out)
			durations[name] = append(durations[name], ns)
			raw, _ := json.Marshal(map[string]any{"block": block, "kernel": name, "wall_ns": ns, "bit_exact": true})
			t.Log("REGTILE_ENCODER_SAMPLE " + string(raw))
		}
	}
	medians := map[string]float64{}
	for name, ns := range durations {
		sort.Slice(ns, func(i, j int) bool { return ns[i] < ns[j] })
		medians[name] = float64(ns[1]+ns[2]) / 2
	}
	raw, _ := json.Marshal(map[string]any{"medians_ns": medians, "speedup": medians["baseline"] / medians["candidate"], "sameweights_pcm_shape": true, "samples_per_kernel": 4, "includes_upload_download": true})
	t.Log("REGTILE_ENCODER_TIMING " + string(raw))
	// Compare candidate public speech against the committed baseline outputs; all
	// tokens/timestamps must match, not merely a zero-WER word normalisation.
	baselineLog := filepath.Join("..", "..", "benchmarks", "speech-foundations", "vulkan-turbo-20260912", "metrics.json")
	data := pinnedSpeechFile(t, baselineLog, "0d1042974f840dc7a30416fd2b60b5a2dd2a69e1a556fcf5063e117154063ac8")
	var old []struct {
		Windows []struct {
			Fixture string           `json:"fixture"`
			Window  WindowTranscript `json:"window"`
		} `json:"windows"`
	}
	if err := json.Unmarshal(data, &old); err != nil || len(old) != 3 {
		t.Fatal("baselineevidence", err)
	}
	expected := map[string]WindowTranscript{}
	for _, w := range old[0].Windows {
		expected[w.Fixture] = w.Window
	}
	if len(expected) != 4 {
		t.Fatal("incomplete baseline fixtures")
	}
	for _, run := range old[1:] {
		for _, w := range run.Windows {
			if !reflect.DeepEqual(expected[w.Fixture], w.Window) {
				t.Fatal("baseline repeats differ")
			}
		}
	}
	for _, p := range pairs {
		r, err := media.OpenCanonicalPCM(ctx, p.decoded.Path)
		if err != nil {
			t.Fatal(err)
		}
		var windows []WindowTranscript
		start := time.Now()
		err = model.TranscribePCMWindows(ctx, r, p.fixture.Samples, tok, PCMTranscribeOptions{Language: p.fixture.Language, Generation: policy, MaxNewTokens: 96, VulkanEncoder: candidate}, func(w WindowTranscript) error { windows = append(windows, w); return nil })
		ns := time.Since(start).Nanoseconds()
		closeErr := r.Close()
		if err != nil || closeErr != nil {
			t.Fatal(err, closeErr)
		}
		if len(windows) != 1 || !reflect.DeepEqual(windows[0], expected[p.fixture.Name]) {
			t.Fatal("candidate transcript regression", p.fixture.Name, windows, expected[p.fixture.Name])
		}
		raw, _ := json.Marshal(map[string]any{"fixture": p.fixture.Name, "wall_ns": ns, "baseline_tokens_timestamps_exact": true, "window": windows[0]})
		t.Log("REGTILE_SPEECH " + string(raw))
	}
	t.Log(fmt.Sprintf("REGTILE_RESOURCES native_bytes=%d arenas=%d", vk.VulkanMemoryStats().Bytes, vk.VulkanMemoryStats().Allocations))
}
