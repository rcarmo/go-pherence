package whisper

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
)

// Reference executes only scalar Go functions, never accelerated model dispatch.
// Attention is independent full-score float64 with explicit F32 materialisation.
func vulkanScalarAttention(q, k, v []float32, rows, heads, dim int) []float32 {
	width := heads * dim
	out := make([]float32, len(q))
	scores := make([]float64, rows)
	for r := 0; r < rows; r++ {
		for h := 0; h < heads; h++ {
			mx := -math.MaxFloat64
			for j := 0; j < rows; j++ {
				sum := 0.
				for d := 0; d < dim; d++ {
					sum += float64(q[r*width+h*dim+d]) * float64(k[j*width+h*dim+d])
				}
				scores[j] = sum / math.Sqrt(float64(dim))
				mx = math.Max(mx, scores[j])
			}
			sum := 0.
			for j := range scores {
				scores[j] = math.Exp(scores[j] - mx)
				sum += scores[j]
			}
			for d := 0; d < dim; d++ {
				value := 0.
				for j := 0; j < rows; j++ {
					value += scores[j] / sum * float64(v[j*width+h*dim+d])
				}
				out[r*width+h*dim+d] = float32(value)
			}
		}
	}
	return out
}
func vulkanScalarBoundaries(enc *Encoder, mel []float32, frames int) [][]float32 {
	c := enc.cfg
	d := c.EncoderDModel
	rows := (frames + 1) / 2
	exactGELU := func(a []float32) {
		for i, x := range a {
			a[i] = float32(.5 * float64(x) * math.Erfc(-float64(x)/math.Sqrt2))
		}
	}
	conv := conv1dForward(mel, enc.Conv1Weight, enc.Conv1Bias, c.NumMelBins, frames, d, 3, 1, 1)
	exactGELU(conv)
	conv = conv1dForward(conv, enc.Conv2Weight, enc.Conv2Bias, d, frames, d, 3, 2, 1)
	exactGELU(conv)
	h := transpose2D(conv, d, rows)
	for i := range h {
		h[i] += enc.PosEmbed[i]
	}
	boundaries := [][]float32{append([]float32(nil), h...)}
	for _, layer := range enc.Layers {
		norm := layerNorm(h, layer.AttnLNWeight, layer.AttnLNBias, rows, d)
		q := linearForwardScalar(norm, layer.QWeight, layer.QBias, rows, d, d)
		k := linearForwardScalar(norm, layer.KWeight, layer.KBias, rows, d, d)
		v := linearForwardScalar(norm, layer.VWeight, layer.VBias, rows, d, d)
		att := vulkanScalarAttention(q, k, v, rows, c.EncoderHeads, c.HeadDim)
		projected := linearForwardScalar(att, layer.OWeight, layer.OBias, rows, d, d)
		for i := range h {
			h[i] += projected[i]
		}
		norm = layerNorm(h, layer.MLPLNWeight, layer.MLPLNBias, rows, d)
		hidden := linearForwardScalar(norm, layer.FC1Weight, layer.FC1Bias, rows, d, c.EncoderFFNDim)
		exactGELU(hidden)
		out := linearForwardScalar(hidden, layer.FC2Weight, layer.FC2Bias, rows, c.EncoderFFNDim, d)
		for i := range h {
			h[i] += out[i]
		}
		boundaries = append(boundaries, append([]float32(nil), h...))
	}
	h = layerNorm(h, enc.FinalLNWeight, enc.FinalLNBias, rows, d)
	return append(boundaries, h)
}
func nativeEncoderCompare(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatal("length", name)
	}
	max := 0.
	for i, v := range got {
		err := math.Abs(float64(v) - float64(want[i]))
		max = math.Max(max, err)
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || err > 1e-4+1e-4*math.Abs(float64(want[i])) {
			t.Fatal(name, i, v, want[i], err)
		}
	}
	b, _ := json.Marshal(map[string]any{"name": name, "values": len(got), "max_abs": max, "abs_budget": 1e-4, "rel_budget": 1e-4})
	t.Log("ENCODER_METRIC " + string(b))
}
func nativeEncoderMemory(t *testing.T, before vk.VulkanMemoryUsage) {
	t.Helper()
	after := vk.VulkanMemoryStats()
	if after.Allocations != before.Allocations || after.Bytes != before.Bytes {
		t.Fatal("encoder memory leak", before, after)
	}
}
func TestVulkanEncoderNative(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_ENCODER") != "1" {
		t.Skip("explicit native encoder compute window required")
	}
	deadline, ok := t.Deadline()
	if !ok || time.Until(deadline) > 3*time.Minute {
		t.Fatal("use go test timeout<=3m")
	}
	if !vk.VulkanInit() {
		t.Fatal("Vulkan unavailable")
	}
	device := vk.VulkanDeviceName()
	want := os.Getenv("GO_PHERENCE_VULKAN_DEVICE")
	if want == "" || !strings.Contains(device, want) || strings.Contains(strings.ToLower(device), "llvmpipe") {
		t.Fatal("set expected physical device name", device)
	}
	limits, err := vk.VulkanLimits()
	if err != nil {
		t.Fatal(err)
	}
	metadata, _ := json.Marshal(map[string]any{"device": device, "limits": limits})
	t.Log(string(metadata))
	before := vk.VulkanMemoryStats()
	configs := []Config{vulkanToyConfig(), {NumMelBins: 80, MaxLength: 34, EncoderLayers: 2, EncoderDModel: 32, EncoderHeads: 2, HeadDim: 16, EncoderFFNDim: 63}, {NumMelBins: 128, MaxLength: 66, EncoderLayers: 4, EncoderDModel: 64, EncoderHeads: 1, HeadDim: 64, EncoderFFNDim: 128}}
	for index, c := range configs {
		for _, frames := range []int{c.MaxLength - 1, c.MaxLength} {
			if !t.Run(fmt.Sprintf("shape%d/frames%d", index, frames), func(t *testing.T) {
				source := vulkanToyEncoder(t, c)
				mel := make([]float32, c.NumMelBins*frames)
				for i := range mel {
					mel[i] = float32(math.Sin(float64(i) * .13))
				}
				refs := vulkanScalarBoundaries(source, mel, frames)
				ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
				defer cancel()
				enc, err := NewVulkanEncoder(ctx, source, frames)
				if enc != nil {
					t.Cleanup(func() {
						if err := enc.Close(); err != nil {
							t.Error(err)
						}
					})
				}
				if err != nil {
					t.Fatal(err)
				}
				if enc.Stats().Stages != 6+12*c.EncoderLayers || enc.Stats().Plans != c.EncoderLayers+2 {
					t.Fatal("stats", enc.Stats())
				}
				stats, _ := json.Marshal(enc.Stats())
				t.Log("ENCODER_STATS " + string(stats))
				// Diagnostics transfer a boundary after each plan, separate from Forward.
				if err := enc.s.input.Upload(ctx, mel); err != nil {
					t.Fatal(err)
				}
				for i, p := range enc.s.plans {
					if err := p.Run(ctx); err != nil {
						t.Fatal(err)
					}
					got := make([]float32, len(refs[i]))
					if err := enc.s.output.Download(ctx, got); err != nil {
						t.Fatal(err)
					}
					nativeEncoderCompare(t, fmt.Sprintf("boundary%d", i), got, refs[i])
				}
				var first []float32
				for repeat := 0; repeat < 3; repeat++ {
					out, err := enc.Forward(ctx, mel)
					if err != nil {
						t.Fatal(err)
					}
					nativeEncoderCompare(t, fmt.Sprintf("forward%d", repeat), out, refs[len(refs)-1])
					if repeat == 0 {
						first = out
					} else {
						equalContextFloats(t, first, out)
					}
				}
				// Device ownership: overwrite every source tensor after upload, including
				// positional weights; completed encoder must no longer depend on host data.
				description, err := describeVulkanEncoder(context.Background(), source, frames)
				if err != nil {
					t.Fatal(err)
				}
				for _, g := range description.weights {
					for _, s := range g {
						for i := range s.data {
							s.data[i] = 0
						}
					}
				}
				source.cfg = Config{}
				out, err := enc.Forward(ctx, mel)
				if err != nil {
					t.Fatal(err)
				}
				equalContextFloats(t, first, out)
				if _, err := enc.Forward(ctx, mel[:len(mel)-1]); err == nil {
					t.Fatal("bad mel extent")
				}
				bad := append([]float32(nil), mel...)
				bad[0] = float32(math.NaN())
				if _, err := enc.Forward(ctx, bad); err == nil {
					t.Fatal("nonfinite mel")
				}
				canceled, stop := context.WithCancel(ctx)
				stop()
				if _, err := enc.Forward(canceled, mel); !errors.Is(err, context.Canceled) {
					t.Fatal("cancelled forward", err)
				}
				// Deterministic cancellation through the full API/backend call
				// chain. A timeout-retained submission is drained explicitly;
				// device-loss/quarantine errors are never retried by this test.
				if index == 0 && frames == c.MaxLength-1 {
					counter := newCheckpointContext(0)
					_, err := enc.Forward(counter, mel)
					counter.cancel()
					if err != nil {
						t.Fatal(err)
					}
					total := counter.calls
					for _, at := range []int{total / 4, total / 2, total - 4} {
						fault := newCheckpointContext(at)
						_, err := enc.Forward(fault, mel)
						fault.cancel()
						if !errors.Is(err, context.Canceled) {
							t.Fatal("fault did not cancel", at, total, err)
						}
						retained := errors.Is(err, vk.ErrVulkanInFlight)
						if !vk.VulkanReady() {
							if !retained {
								t.Fatal("unexpected nonready state", err)
							}
							drainCtx, drainCancel := context.WithTimeout(context.Background(), time.Second)
							drainErr := vk.VulkanDrain(drainCtx, time.Second)
							drainCancel()
							if drainErr != nil {
								t.Fatal(drainErr)
							}
						}
						out, err := enc.Forward(ctx, mel)
						if err != nil {
							t.Fatal("reuse after cancellation", err)
						}
						equalContextFloats(t, first, out)
						t.Logf("ENCODER_CANCEL at=%d total=%d retained=%t reuse_bit_exact=true", at, total, retained)
					}
				}
				// Rejects leave a healthy encoder reusable; copies share the same owner.
				copy := *enc
				out, err = copy.Forward(ctx, mel)
				if err != nil {
					t.Fatal(err)
				}
				equalContextFloats(t, first, out)
				if err := copy.Close(); err != nil {
					t.Fatal(err)
				}
				if _, err := enc.Forward(ctx, mel); !errors.Is(err, vk.ErrVulkanClosed) {
					t.Fatal("closed forward", err)
				}
			}) {
				return
			}
			nativeEncoderMemory(t, before)
		}
	}
	// Force the second weight arena allocation to fail after the first succeeds;
	// constructors/plans must unwind all resources without changing global budget.
	if !t.Run("allocation-rollback", func(t *testing.T) {
		source := vulkanToyEncoder(t, vulkanToyConfig())
		if err := vk.VulkanSetMemoryBudget(vk.VulkanMemoryBudget{MaxAllocations: before.Allocations + 1}); err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := vk.VulkanSetMemoryBudget(before.Budget); err != nil {
				t.Error(err)
			}
		}()
		enc, err := NewVulkanEncoder(context.Background(), source, 17)
		if enc != nil {
			defer enc.Close()
		}
		if err == nil || enc != nil {
			t.Fatal("failedallocation rollback", enc, err)
		}
		nativeEncoderMemory(t, before)
	}) {
		return
	}
	nativeEncoderMemory(t, before)
	if !t.Run("PCM-bridge", nativePCMVulkanBridge) {
		return
	}
	nativeEncoderMemory(t, before)
}
