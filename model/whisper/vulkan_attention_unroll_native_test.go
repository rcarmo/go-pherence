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

func TestWhisperVulkanAttentionUnrollNative(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_WHISPER_ATTN_UNROLL") != "1" {
		t.Skip("explicit exact attention unroll")
	}
	d, ok := t.Deadline()
	if !ok || time.Until(d) > 120*time.Second {
		t.Fatal("bounded120s")
	}
	ctx, cancel := context.WithDeadline(context.Background(), d.Add(-time.Second))
	defer cancel()
	model, _, _, file := pinnedLegacyWhisperModelOpen(t, ctx)
	defer file.Close()
	if !vk.VulkanInit() || os.Getenv("GO_PHERENCE_VULKAN_DEVICE") == "" || !strings.Contains(vk.VulkanDeviceName(), os.Getenv("GO_PHERENCE_VULKAN_DEVICE")) {
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
	baseline, e := NewVulkanEncoderOriginalQ5MLP(ctx, model.Encoder, model.Config.MaxLength, file)
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
	op, e := NewVulkanEncoderOriginalQ5AttentionUnroll4(ctx, model.Encoder, model.Config.MaxLength, file)
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
	out, e := op.Forward(ctx, mel)
	if e != nil {
		t.Fatal(e)
	}
	check(out)
	c := newCheckpointContext(0)
	out, e = op.Forward(c, mel)
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
	t.Logf("UNROLL_HIDDEN values=%d checkpoints=%d bitExact=true statsUnchanged=true cancelReuse=true", len(ref), c.calls)
}

func TestWhisperVulkanAttentionUnrollAdmission(t *testing.T) {
	source := vulkanToyEncoder(t, vulkanToyConfig())
	if e, err := NewVulkanEncoderOriginalQ5AttentionUnroll4(nil, source, 17, nil); e != nil || err == nil {
		t.Fatal("nil")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if e, err := NewVulkanEncoderOriginalQ5AttentionUnroll4(ctx, source, 17, nil); e != nil || !errors.Is(err, context.Canceled) {
		t.Fatal("cancel", err)
	}
	if e, err := NewVulkanEncoderOriginalQ5AttentionUnroll4(context.Background(), source, 17, nil); e != nil || err == nil {
		t.Fatal("small head dimension")
	}
	if vulkanDefaultLinearMode != vulkanLinearF32RegTile {
		t.Fatal("default")
	}
}
