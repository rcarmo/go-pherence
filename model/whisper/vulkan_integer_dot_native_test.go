package whisper

import (
	"context"
	"errors"
	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

// Experimental arithmetic diagnostic: hidden drift is measured, never accepted
// through the F32 exactness gate. This test does not qualify acoustic accuracy.
func TestVulkanIntegerDotEncoderNative(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_WHISPER_INTDOT_ENCODER") != "1" {
		t.Skip("explicit integer-dot encoder diagnostic")
	}
	d, ok := t.Deadline()
	if !ok || time.Until(d) > 120*time.Second {
		t.Fatal("bounded120s")
	}
	ctx, cancel := context.WithDeadline(context.Background(), d.Add(-time.Second))
	defer cancel()
	model, _, _, file := pinnedLegacyWhisperModelOpen(t, ctx)
	defer file.Close()
	if !vk.VulkanInitIntegerDot() || os.Getenv("GO_PHERENCE_VULKAN_DEVICE") == "" || !strings.Contains(vk.VulkanDeviceName(), os.Getenv("GO_PHERENCE_VULKAN_DEVICE")) {
		t.Fatal("physical integer-dot device")
	}
	before := vk.VulkanMemoryStats()
	if before.Bytes != 0 || before.Allocations != 0 {
		t.Fatal("isolated")
	}
	if err := vk.VulkanSetMemoryBudget(vk.VulkanMemoryBudget{MaxBytes: 4 << 30, MaxAllocations: 40}); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := vk.VulkanSetMemoryBudget(before.Budget); err != nil {
			t.Error(err)
		}
	}()
	mel := make([]float32, model.Config.NumMelBins*model.Config.MaxLength)
	for i := range mel {
		mel[i] = float32(i%17-8) / 32
	}
	base, e := NewVulkanEncoderOriginalQ5MLP(ctx, model.Encoder, model.Config.MaxLength, file)
	if e != nil {
		t.Fatal(e)
	}
	want, e := base.Forward(ctx, mel)
	if e != nil {
		t.Fatal(e)
	}
	stats := base.Stats()
	if e = base.Close(); e != nil {
		t.Fatal(e)
	}
	candidate, e := NewVulkanEncoderOriginalQ5IntegerDotMLP(ctx, model.Encoder, model.Config.MaxLength, file)
	if e != nil {
		t.Fatal(e)
	}
	defer candidate.Close()
	got, e := candidate.Forward(ctx, mel)
	if e != nil {
		t.Fatal(e)
	}
	different := 0
	var maximum, total float64
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("finite")
		}
		if math.Float32bits(v) != math.Float32bits(want[i]) {
			different++
		}
		delta := math.Abs(float64(v - want[i]))
		if delta > maximum {
			maximum = delta
		}
		total += delta
	}
	cs := candidate.Stats()
	if cs.WeightBytes != stats.WeightBytes || cs.ScratchBytes <= stats.ScratchBytes || cs.Stages != stats.Stages+model.Config.EncoderLayers*2 || cs.Plans != stats.Plans {
		t.Fatal("stats", stats, cs)
	}
	countContext := newCheckpointContext(0)
	_, e = candidate.Forward(countContext, mel)
	calls := countContext.calls
	if e != nil {
		t.Fatal(e)
	}
	for _, at := range []int{1, calls / 2, calls - 2} {
		c := newCheckpointContext(at)
		_, e = candidate.Forward(c, mel)
		if !errors.Is(e, context.Canceled) {
			t.Fatal("cancel", at, e)
		}
		if vk.VulkanMemoryStats().InFlight {
			if e = vk.VulkanDrain(ctx, time.Second); e != nil {
				t.Fatal(e)
			}
		}
		reuse, e := candidate.Forward(ctx, mel)
		if e != nil {
			t.Fatal(e)
		}
		for i, v := range reuse {
			if math.Float32bits(v) != math.Float32bits(got[i]) {
				t.Fatal("reuse", i)
			}
		}
	}
	t.Logf("INTDOT_HIDDEN values=%d different=%d maxAbs=%.9g meanAbs=%.9g checkpoints=%d scratchAdded=%d", len(got), different, maximum, total/float64(len(got)), calls, cs.ScratchBytes-stats.ScratchBytes)
	if e = candidate.Close(); e != nil {
		t.Fatal(e)
	}
	after := vk.VulkanMemoryStats()
	if after.Bytes != before.Bytes || after.Allocations != before.Allocations || after.InFlight {
		t.Fatal("native leak", before, after)
	}
}
func TestVulkanIntegerDotEncoderRejectsBeforeDevice(t *testing.T) {
	source := vulkanToyEncoder(t, vulkanToyConfig())
	if e, err := NewVulkanEncoderOriginalQ5IntegerDotMLP(nil, source, 17, nil); e != nil || err == nil {
		t.Fatal("nil")
	}
	if e, err := NewVulkanEncoderOriginalQ5IntegerDotMLP(context.Background(), nil, 17, nil); e != nil || err == nil {
		t.Fatal("source")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if e, err := NewVulkanEncoderOriginalQ5IntegerDotMLP(ctx, source, 17, nil); e != nil || !errors.Is(err, context.Canceled) {
		t.Fatal("cancel", err)
	}
	if vulkanDefaultLinearMode != vulkanLinearF32RegTile {
		t.Fatal("default")
	}
}
