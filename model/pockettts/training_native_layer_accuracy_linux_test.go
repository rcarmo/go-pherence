//go:build linux

package pockettts

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"github.com/rcarmo/go-pherence/backends/vulkan"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

// Isolate each layer at independent CPU inputs. Reference arrays are never
// supplied to candidate execution except the admitted layer source sequence.
func TestPocketNativeLayerAccuracy(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_LAYER_ACCURACY_NATIVE") != "1" {
		t.Skip("explicit per-layer arithmetic qualification")
	}
	lm, batch := nativeConditionerInputs(t)
	sequence, _, err := trunkCaptureSequence(lm, batch)
	if err != nil {
		t.Fatal(err)
	}
	rows := len(sequence) / lm.Hidden
	refTape, _ := lm.Transformer.forwardTraining(rows, sequence)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	if !vulkan.VulkanInit() || vulkan.VulkanDeviceName() != "NVIDIA GeForce RTX 3060" {
		t.Fatal("runtime")
	}
	if e := vulkan.VulkanSetMemoryBudget(vulkan.VulkanMemoryBudget{MaxBytes: 2 << 30, MaxAllocations: 4096}); e != nil {
		t.Fatal(e)
	}
	op, e := vulkan.NewVkTrainingF32(ctx)
	if e != nil {
		t.Fatal(e)
	}
	type boundary struct {
		Layer int
		Name  string
		Error reverseError
	}
	var report []boundary
	type softmaxRow struct {
		Layer, Head, Query        int
		Scores, Native, Reference []float32
	}
	var softmaxRows []softmaxRow
	input := sequence
	for index, layer := range lm.Transformer.Layers {
		cpu := refTape.layers[index]
		tr := *lm.Transformer
		tr.Layers = []TransformerLayerCPU{layer}
		tr.FinalWeight = nil
		tr.FinalBias = nil
		p, e := tr.buildTransformerTrainingTape(input, make([]float32, len(input)), rows, 2<<30, true)
		if e != nil {
			t.Fatal(e)
		}
		refs := [][]float32{cpu.norm1.normalized, linearForwardRowsTraining(layer.InProjection, cpu.norm1.normalized, rows), cpu.q, cpu.k, cpu.probabilities, cpu.attention, cpu.projected, cpu.norm2.normalized, cpu.fc1Pre, cpu.fc1, cpu.down, cpu.output}
		labels := []string{"norm1", "qkv", "q-rope", "k-rope", "softmax", "attention", "projection", "norm2", "fc1", "gelu", "fc2", "output"}
		statsRefs := [][]float32{}
		for _, nt := range []transformerNormTape{cpu.norm1, cpu.norm2} {
			s := make([]float32, rows*5)
			for r := 0; r < rows; r++ {
				var mean, variance float32
				v := nt.input[r*tr.Width : (r+1)*tr.Width]
				for _, x := range v {
					mean += x
				}
				mean /= float32(tr.Width)
				for _, x := range v {
					d := x - mean
					variance += d * d
				}
				variance /= float32(tr.Width)
				s[r*5] = mean
				s[r*5+2] = nt.inv[r]
			}
			statsRefs = append(statsRefs, s)
		}
		var statIDs []int
		ids := []int{}
		normCount, linearCount := 0, 0
		for _, c := range p.Tape.Commands {
			take := false
			switch c.Op {
			case "norm-forward":
				if c.Mode == 0 {
					statIDs = append(statIDs, c.Output)
				}
				if c.Mode == 1 {
					take = true
					normCount++
				}
			case "linear-forward":
				if c.Mode == 0 {
					take = true
					linearCount++
				}
			case "rope":
				if len(ids) < 4 {
					take = true
				}
			case "attention-forward":
				if c.Mode == 1 || c.Mode == 2 {
					take = true
				}
			case "gelu":
				if c.Mode == 0 {
					take = true
				}
			}
			if take {
				ids = append(ids, c.Output)
			}
			if linearCount == 4 {
				break
			}
		}
		ids = append(ids, p.Tape.Outputs[0])
		if len(ids) != len(refs) || normCount != 2 {
			t.Fatal("layer roots", len(ids), len(refs), normCount)
		}
		ids = append(ids, statIDs...)
		refs = append(refs, statsRefs...)
		labels = append(labels, "norm1-stats", "norm2-stats")
		p.Tape.Outputs = ids
		tape, _, e := train.PruneTape(p.Tape, ids, 2<<30)
		if e != nil {
			t.Fatal(e)
		}
		if rss, e := flowCaptureRSS(); e != nil || rss > 6<<30 {
			t.Fatal("host cap", rss, e)
		}
		owner, e := vulkan.NewVkTrainingTape(ctx, op, tape, 2<<30)
		if e != nil {
			t.Fatal(e)
		}
		got, e := owner.Run(ctx)
		if e != nil {
			t.Fatal("native failure no retry", e)
		}
		for h := 0; h < tr.Heads; h++ {
			for q := 0; q < rows; q++ {
				base := (h*rows + q) * rows
				if !slices.Equal(got[4][base:base+rows], cpu.probabilities[base:base+rows]) {
					scores := make([]float32, q+1)
					scale := float32(1 / math.Sqrt(float64(tr.HeadDim)))
					for key := range scores {
						off := h * tr.HeadDim
						scores[key] = dotScalar(cpu.q[q*tr.Width+off:q*tr.Width+off+tr.HeadDim], cpu.k[key*tr.Width+off:key*tr.Width+off+tr.HeadDim]) * scale
					}
					softmaxRows = append(softmaxRows, softmaxRow{index, h, q, scores, append([]float32(nil), got[4][base:base+q+1]...), append([]float32(nil), cpu.probabilities[base:base+q+1]...)})
				}
			}
		}
		for i, v := range got {
			d, e := reverseCompare(v, refs[i])
			if e != nil {
				t.Fatal(labels[i], e)
			}
			if i >= 12 {
				for r := 0; r < rows; r++ {
					if v[r*5] != refs[i][r*5] || v[r*5+2] != refs[i][r*5+2] {
						t.Logf("layer%d %s row%d mean %g/%g inv %g/%g", index, labels[i], r, v[r*5], refs[i][r*5], v[r*5+2], refs[i][r*5+2])
					}
				}
			}
			report = append(report, boundary{index, labels[i], d})
			t.Logf("layer%d %s max=%g RMS=%g", index, labels[i], d.Max, d.RMS)
		}
		if e = owner.Close(); e != nil {
			t.Fatal(e)
		}
		input = cpu.output
	}
	if e = op.Close(); e != nil {
		t.Fatal(e)
	}
	u := vulkan.VulkanMemoryStats()
	if u.Bytes != 0 || u.Allocations != 0 || u.InFlight || u.Uncertain || u.DeviceLost {
		t.Fatal("closure", u)
	}
	if e = flowCaptureWrite(filepath.Join(os.Getenv("GO_PHERENCE_POCKET_LAYER_ACCURACY_OUTPUT"), "softmax-rows.json"), softmaxRows); e != nil {
		t.Fatal(e)
	}
	if e = flowCaptureWrite(filepath.Join(os.Getenv("GO_PHERENCE_POCKET_LAYER_ACCURACY_OUTPUT"), "layer-accuracy.json"), report); e != nil {
		t.Fatal(fmt.Errorf("report: %w", e))
	}
}
