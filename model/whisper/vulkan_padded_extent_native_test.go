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

func TestWhisperPaddedExtentNative(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_WHISPER_PADDED_EXTENT") != "1" {
		t.Skip("explicit paddedhidden")
	}
	d, ok := t.Deadline()
	if !ok || time.Until(d) > 120*time.Second {
		t.Fatal("120s")
	}
	ctx, cancel := context.WithDeadline(context.Background(), d.Add(-time.Second))
	defer cancel()
	model, _, _, file := pinnedLegacyWhisperModelMode(t, ctx, true)
	defer file.Close()
	if !vk.VulkanInit() || os.Getenv("GO_PHERENCE_VULKAN_DEVICE") == "" || !strings.Contains(vk.VulkanDeviceName(), os.Getenv("GO_PHERENCE_VULKAN_DEVICE")) {
		t.Fatal("physical")
	}
	before := vk.VulkanMemoryStats()
	if before.Bytes != 0 {
		t.Fatal("isolated")
	}
	// Explicit copy oracle independently appends36 physical zeros per layer.
	code, e := os.ReadFile(os.Getenv("GO_PHERENCE_TEST_PAD_COPY_SPV"))
	if e != nil {
		t.Fatal(e)
	}
	copyKernel, e := vk.VkKernelCreate(code, 2, 8)
	if e != nil {
		t.Fatal(e)
	}
	defer copyKernel.Close()
	arena, e := vk.NewVkTensorArena(ctx, 2*1536*1280*4+4096)
	if e != nil {
		t.Fatal(e)
	}
	defer arena.Close()
	pk, e := arena.AllocF32(ctx, 1536, 1280)
	if e != nil {
		t.Fatal(e)
	}
	pv, e := arena.AllocF32(ctx, 1536, 1280)
	if e != nil {
		t.Fatal(e)
	}
	factory := func(c context.Context, s []vk.VkF32Stage) (*vk.VkF32Plan, error) {
		if len(s) == 12 {
			attn := s[4]
			attn.Tensors = append([]*vk.VkTensorF32(nil), attn.Tensors...)
			copies := make([]vk.VkF32Stage, 2)
			for i, out := range []*vk.VkTensorF32{pk, pv} {
				copies[i] = vk.VkF32Stage{Kernel: copyKernel, Groups: [3]uint32{(1536*1280 + 255) / 256, 1, 1}, Tensors: []*vk.VkTensorF32{attn.Tensors[i+1], out}, PushWords: []uint32{1500 * 1280, 1536 * 1280}}
				attn.Tensors[i+1] = out
			}
			attn.PushWords = append([]uint32(nil), attn.PushWords...)
			attn.PushWords[1] = 1536
			next := append([]vk.VkF32Stage(nil), s[:4]...)
			next = append(next, copies...)
			next = append(next, attn)
			s = append(next, s[5:]...)
		}
		return vk.NewVkF32Plan(c, s)
	}
	mel := make([]float32, 128*3000)
	for i := range mel {
		mel[i] = float32(i%17-8) / 32
	}
	reference, e := newVulkanEncoderPackedOnlyMode(ctx, model.Encoder, 3000, factory, vulkanLinearOriginalQ5AttentionOutputILP, file, true)
	if e != nil {
		t.Fatal(e)
	}
	ref, e := reference.Forward(ctx, mel)
	if e != nil {
		t.Fatal(e)
	}
	bs := reference.Stats()
	if e := reference.Close(); e != nil {
		t.Fatal(e)
	}
	if e := arena.Close(); e != nil {
		t.Fatal(e)
	}
	if e := copyKernel.Close(); e != nil {
		t.Fatal(e)
	}
	op, e := newVulkanEncoderPackedOnlyMode(ctx, model.Encoder, 3000, vk.NewVkF32Plan, vulkanLinearOriginalQ5PaddedKeyExtent, file, true)
	if e != nil {
		t.Fatal(e)
	}
	defer op.Close()
	if op.Stats() != bs {
		t.Fatal("stats", op.Stats(), bs)
	}
	check := func(got []float32) {
		if len(got) != len(ref) {
			t.Fatal("extent")
		}
		for i, v := range got {
			if math.Float32bits(v) != math.Float32bits(ref[i]) {
				t.Fatal("physicalpadbits", i)
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
			if e := vk.VulkanDrain(ctx, time.Second); e != nil {
				t.Fatal(e)
			}
		}
		out, e = op.Forward(ctx, mel)
		if e != nil {
			t.Fatal(e)
		}
		check(out)
	}
	if e := op.Close(); e != nil {
		t.Fatal(e)
	}
	after := vk.VulkanMemoryStats()
	if after.Bytes != before.Bytes || after.Allocations != before.Allocations || after.InFlight {
		t.Fatal("leak", after)
	}
	t.Logf("PADDED_EXTENT_PHYSICAL_HIDDEN values=%d checkpoints=%d bitsExact=true statsUnchanged=true cancelReuse=true", len(ref), c.calls)
}
func TestWhisperPaddedExtentAdmission(t *testing.T) {
	source := vulkanToyEncoder(t, vulkanToyConfig())
	if e, err := NewVulkanEncoderOriginalQ5PaddedKeyExtent(nil, source, 17, nil); e != nil || err == nil {
		t.Fatal("nil")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if e, err := NewVulkanEncoderOriginalQ5PaddedKeyExtent(ctx, source, 17, nil); e != nil || !errors.Is(err, context.Canceled) {
		t.Fatal("cancel")
	}
	if e, err := NewVulkanEncoderOriginalQ5PaddedKeyExtent(context.Background(), source, 17, nil); e != nil || err == nil {
		t.Fatal("originaladmission")
	}
	if vulkanDefaultLinearMode != vulkanLinearF32RegTile {
		t.Fatal("default")
	}
}
