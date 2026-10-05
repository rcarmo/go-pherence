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

// The MMQ schedule must leave the complete padded integer-dot encoder bit-identical.
func TestWhisperVulkanIntegerDotMMQNative(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_WHISPER_INTDOT_MMQ") != "1" {
		t.Skip("explicit integer-dot MMQ encoder comparison")
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
		t.Fatal("physicaldevice")
	}
	before := vk.VulkanMemoryStats()
	if before.Bytes != 0 {
		t.Fatal("isolated")
	}
	mel := make([]float32, model.Config.NumMelBins*model.Config.MaxLength)
	for i := range mel {
		mel[i] = float32(i%17-8) / 32
	}
	baseline, e := newVulkanEncoderPackedMode(ctx, model.Encoder, model.Config.MaxLength, vk.NewVkF32Plan, vulkanLinearOriginalQ5PaddedIntegerDotAll, file)
	if e != nil {
		t.Fatal(e)
	}
	ref, e := baseline.Forward(ctx, mel)
	if e != nil {
		t.Fatal(e)
	}
	bs := baseline.Stats()
	if e = baseline.Close(); e != nil {
		t.Fatal(e)
	}
	op, e := newVulkanEncoderPackedMode(ctx, model.Encoder, model.Config.MaxLength, vk.NewVkF32Plan, vulkanLinearOriginalQ5PaddedIntegerDotMMQ, file)
	if e != nil {
		t.Fatal(e)
	}
	defer op.Close()
	if op.Stats() != bs {
		t.Fatal("storage/geometry changed", bs, op.Stats())
	}
	check := func(got []float32) {
		if len(got) != len(ref) {
			t.Fatal("extent")
		}
		for i, v := range got {
			if math.Float32bits(v) != math.Float32bits(ref[i]) {
				t.Fatal("hiddenbits", i, v, ref[i])
			}
		}
	}
	c := newCheckpointContext(0)
	out, e := op.Forward(c, mel)
	if e != nil {
		t.Fatal(e)
	}
	check(out)
	for _, at := range []int{1, c.calls / 2, c.calls - 4} {
		_, e = op.Forward(newCheckpointContext(at), mel)
		if !errors.Is(e, context.Canceled) {
			t.Fatal("cancel", at, e)
		}
		if !vk.VulkanReady() {
			if e = vk.VulkanDrain(ctx, time.Second); e != nil {
				t.Fatal(e)
			}
		}
		out, e = op.Forward(ctx, mel)
		if e != nil {
			t.Fatal(e)
		}
		check(out)
	}
	if e = op.Close(); e != nil {
		t.Fatal(e)
	}
	after := vk.VulkanMemoryStats()
	if after.Bytes != before.Bytes || after.Allocations != before.Allocations || after.InFlight {
		t.Fatal("leak")
	}
	t.Logf("INTDOT_MMQ_HIDDEN values=%d checkpoints=%d bitExact=true statsUnchanged=true cancelReuse=true", len(ref), c.calls)
}
