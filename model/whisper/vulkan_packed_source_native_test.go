package whisper

import (
	"context"
	"errors"
	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	legacy "github.com/rcarmo/go-pherence/loader/whisperggml"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

// Trained original-Q5 encoder admission, hidden parity, cancel/drain/reuse.
// Single owner at a time keeps native memory inside the unchanged cap.
func TestVulkanOriginalQ5SourceNative(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_PACKED_SOURCE") != "1" {
		t.Skip("explicit bounded original-Q5 graph")
	}
	deadline, ok := t.Deadline()
	if !ok || time.Until(deadline) > 300*time.Second {
		t.Fatal("bounded300s")
	}
	ctx, stop := context.WithDeadline(context.Background(), deadline.Add(-time.Second))
	defer stop()
	model, _, _ := pinnedLegacyWhisperModel(t, ctx)
	if !vk.VulkanInit() || os.Getenv("GO_PHERENCE_VULKAN_DEVICE") == "" || !strings.Contains(vk.VulkanDeviceName(), os.Getenv("GO_PHERENCE_VULKAN_DEVICE")) {
		t.Fatal("device")
	}
	before := vk.VulkanMemoryStats()
	defer func() {
		after := vk.VulkanMemoryStats()
		if after.Bytes != before.Bytes || after.Allocations != before.Allocations || after.InFlight {
			t.Error("leak", before, after)
		}
	}()
	mel := make([]float32, model.Config.NumMelBins*model.Config.MaxLength)
	for i := range mel {
		mel[i] = float32(math.Sin(float64(i)*.013) * .7)
	}
	base, err := NewVulkanEncoderTile64Key32ScoreILP(ctx, model.Encoder, model.Config.MaxLength)
	if err != nil {
		if base != nil {
			base.Close()
		}
		t.Fatal(err)
	}
	ref, err := base.Forward(ctx, mel)
	if err != nil {
		base.Close()
		t.Fatal(err)
	}
	baseStats := base.Stats()
	if err = base.Close(); err != nil {
		t.Fatal(err)
	}
	file, err := legacy.Open(ctx, os.Getenv("GO_PHERENCE_WHISPER_GGML"), os.Getenv("GO_PHERENCE_WHISPER_GGML_SHA256"))
	if err != nil {
		t.Fatal(err)
	}
	packedModel, op, err := LoadOriginalQ5ResidentChecked(ctx, file, model.Config)
	file.Close()
	if err != nil {
		if op != nil {
			op.Close()
		}
		t.Fatal(err)
	}
	if packedModel.Encoder != nil || packedModel.validatePCMModel() == nil {
		op.Close()
		t.Fatal("CPU fallback admitted")
	}
	if err := packedModel.ValidatePCMVulkanHostDecoder(ctx, op); err != nil {
		op.Close()
		t.Fatal(err)
	}
	if len(packedModel.Decoder.Layers) != len(model.Decoder.Layers) {
		op.Close()
		t.Fatal("decoder geometry")
	}
	for i, l := range packedModel.Decoder.Layers {
		for j, v := range l.FC1Weight {
			if math.Float32bits(v) != math.Float32bits(model.Decoder.Layers[i].FC1Weight[j]) {
				op.Close()
				t.Fatal("decoder drift", i, j)
			}
		}
	}
	defer func() {
		if err := op.Close(); err != nil {
			t.Error(err)
		}
	}()
	got, err := op.Forward(ctx, mel)
	if err != nil {
		t.Fatal(err)
	}
	exact := func(values []float32) {
		t.Helper()
		if len(values) != len(ref) {
			t.Fatal("extent")
		}
		for i, v := range values {
			if math.Float32bits(v) != math.Float32bits(ref[i]) {
				t.Fatal("hidden drift", i, v, ref[i])
			}
		}
	}
	exact(got)
	counter := newCheckpointContext(0)
	if _, err = op.Forward(counter, mel); err != nil {
		counter.cancel()
		t.Fatal(err)
	}
	calls := counter.calls
	counter.cancel()
	for _, point := range []int{1, calls / 2, calls - 4} {
		fault := newCheckpointContext(point)
		_, err = op.Forward(fault, mel)
		fault.cancel()
		if !errors.Is(err, context.Canceled) {
			t.Fatal("cancel", point, err)
		}
		if !vk.VulkanReady() {
			drainCtx, cancel := context.WithTimeout(ctx, time.Second)
			err = vk.VulkanDrain(drainCtx, time.Second)
			cancel()
			if err != nil {
				t.Fatal(err)
			}
		}
		got, err = op.Forward(ctx, mel)
		if err != nil {
			t.Fatal(err)
		}
		exact(got)
	}
	if op.Stats().WeightBytes >= baseStats.WeightBytes || op.Stats().Stages != baseStats.Stages || op.Stats().Plans != baseStats.Plans {
		t.Fatal("stats", baseStats, op.Stats())
	}
	t.Logf("PACKED_SOURCE exact_hidden=%d checkpoints=%d baseline_weights=%d packed_weights=%d", len(ref), calls, baseStats.WeightBytes, op.Stats().WeightBytes)
}
