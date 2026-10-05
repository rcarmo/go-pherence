//go:build linux

package pockettts

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	train "github.com/rcarmo/go-pherence/backends/training"
	"github.com/rcarmo/go-pherence/backends/vulkan"
)

// Each native stage receives fixed independent CPU boundaries. This localises
// arithmetic drift; reference values are never substituted in a training run.
func TestPocketJointFixedBoundariesNative(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_JOINT_BOUNDARIES_NATIVE") != "1" {
		t.Skip("explicit joint boundary diagnostic")
	}
	root := os.Getenv("GO_PHERENCE_POCKET_JOINT_LAYER_INPUT")
	lm, _, _, _ := releasedTrainingModels(t)
	read := func(name string) []float32 {
		raw, err := os.ReadFile(filepath.Join(root, name))
		if err != nil || len(raw) != 950*lm.Hidden*4 {
			t.Fatal("boundary extent", name, err)
		}
		values := make([]float32, len(raw)/4)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
		if !finiteF32(values) {
			t.Fatal("nonfinite boundary")
		}
		return values
	}
	ctx, cancel := context.WithTimeout(context.Background(), 240*time.Second)
	defer cancel()
	if !vulkan.VulkanInit() || vulkan.VulkanDeviceName() != "NVIDIA GeForce RTX 3060" {
		t.Fatal("no fallback")
	}
	if err := vulkan.VulkanSetMemoryBudget(vulkan.VulkanMemoryBudget{MaxBytes: 2 << 30, MaxAllocations: 4096}); err != nil {
		t.Fatal(err)
	}
	op, err := vulkan.NewVkTrainingF32(ctx)
	if err != nil {
		t.Fatal(err)
	}
	compare := func(name string, got, want []float32) {
		if len(got) != len(want) || !finiteF32(got) {
			t.Fatal("boundary result", name)
		}
		var maxError, square, energy float64
		changed, index := 0, 0
		for i, x := range got {
			d := math.Abs(float64(x) - float64(want[i]))
			square += d * d
			energy += float64(want[i]) * float64(want[i])
			if d != 0 {
				changed++
			}
			if d > maxError {
				maxError, index = d, i
			}
		}
		t.Logf("boundary=%s values=%d changed=%d max=%g index=%d got=%g want=%g rms=%g relativeL2=%g", name, len(got), changed, maxError, index, got[index], want[index], math.Sqrt(square/float64(len(got))), math.Sqrt(square/math.Max(energy, 1e-30)))
	}
	run := func(stage TransformerTrainingTape, roots []int) [][]float32 {
		tape, _, err := train.PruneTape(stage.Tape, roots, 2<<30)
		if err != nil {
			t.Fatal(err)
		}
		owner, err := vulkan.NewVkTrainingTape(ctx, op, tape, 2<<30)
		if err != nil {
			t.Fatal(err)
		}
		values, err := owner.Run(ctx)
		if err != nil {
			t.Fatal("submitted diagnostic, no retry", err)
		}
		if err = owner.Close(); err != nil {
			t.Fatal(err)
		}
		return values
	}
	norm, err := buildTransformerFinalNormStage(lm.Transformer, read("cpu-final-input.f32"), read("cpu-final-seed.f32"), 950, false, 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	values := run(norm, norm.Tape.Outputs[1:2])
	compare("final-dx", values[0], read("cpu-final-dx.f32"))
	for i := len(lm.Transformer.Layers) - 1; i >= 0; i-- {
		if os.Getenv("GO_PHERENCE_POCKET_JOINT_BOUNDARY_DETAIL") == "1" && (i < 2 || i > 4) {
			continue
		}
		stage, err := buildTransformerLayerStage(lm.Transformer, i, read(fmt.Sprintf("cpu-layer%d-sequence.f32", i)), read(fmt.Sprintf("cpu-layer%d-adjoint.f32", i)), 950, false, 2<<30)
		if err != nil {
			t.Fatal(err)
		}
		if os.Getenv("GO_PHERENCE_POCKET_JOINT_BOUNDARY_DETAIL") == "1" {
			input := read(fmt.Sprintf("cpu-layer%d-sequence.f32", i))
			l := lm.Transformer.Layers[i]
			cpu, _ := l.forwardTraining(950, lm.Hidden, lm.Transformer.Heads, lm.Transformer.HeadDim, lm.Transformer.Context, lm.Transformer.MaxPeriod, input)
			refs := [][]float32{cpu.norm1.normalized, linearForwardRowsTraining(l.InProjection, cpu.norm1.normalized, 950), cpu.q, cpu.k, cpu.probabilities, cpu.attention, cpu.projected, cpu.norm2.input, cpu.norm2.normalized, cpu.fc1Pre, cpu.fc1, cpu.down, cpu.output}
			labels := []string{"norm1", "qkv", "q-rope", "k-rope", "probabilities", "attention", "projected", "residual1", "norm2", "fc1Pre", "gelu", "down", "output"}
			ids := []int{}
			linears := 0
			for _, cmd := range stage.Tape.Commands {
				take := false
				switch cmd.Op {
				case "norm-forward":
					if cmd.Mode == 1 {
						take = true
						if len(ids) > 1 {
							ids = append(ids, cmd.Inputs[0])
						}
					}
				case "linear-forward":
					if cmd.Mode == 0 {
						take = true
						linears++
					}
				case "rope":
					if len(ids) < 4 {
						take = true
					}
				case "attention-forward":
					if cmd.Mode == 1 || cmd.Mode == 2 {
						take = true
					}
				case "gelu":
					if cmd.Mode == 0 {
						take = true
					}
				}
				if take {
					ids = append(ids, cmd.Output)
				}
				if linears == 4 {
					break
				}
			}
			ids = append(ids, stage.Tape.Outputs[0])
			if len(ids) != len(refs) {
				t.Fatal("detail roots", ids)
			}
			stage.Tape.Outputs = append(stage.Tape.Outputs, ids...)
			v := run(stage, ids)
			for j := range v {
				compare(fmt.Sprintf("layer%d-%s", i, labels[j]), v[j], refs[j])
			}
			var points []struct {
				Index              int
				Input, CPU, Native float32
			}
			for j, x := range v[10] {
				if x != refs[10][j] {
					points = append(points, struct {
						Index              int
						Input, CPU, Native float32
					}{j, refs[9][j], refs[10][j], x})
				}
			}
			if output := os.Getenv("GO_PHERENCE_POCKET_JOINT_BOUNDARY_OUTPUT"); output != "" {
				data, err := json.MarshalIndent(points, "", "  ")
				if err != nil {
					t.Fatal(err)
				}
				file, err := os.OpenFile(filepath.Join(output, fmt.Sprintf("gelu-layer%d.json", i)), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
				if err != nil {
					t.Fatal(err)
				}
				_, err = file.Write(data)
				closeErr := file.Close()
				if err != nil || closeErr != nil {
					t.Fatal(err, closeErr)
				}
			}
			continue
		}
		// Forward and reverse execute separately, as in the production coordinator.
		values = run(stage, stage.Tape.Outputs[:1])
		compare(fmt.Sprintf("layer%d-output", i), values[0], read(fmt.Sprintf("cpu-layer%d-output.f32", i)))
		values = run(stage, stage.Tape.Outputs[1:2])
		compare(fmt.Sprintf("layer%d-dx", i), values[0], read(fmt.Sprintf("cpu-layer%d-dx.f32", i)))
	}
	if err = op.Close(); err != nil {
		t.Fatal(err)
	}
	if stats := vulkan.VulkanMemoryStats(); stats.Bytes != 0 || stats.Allocations != 0 || stats.InFlight {
		t.Fatal("boundary diagnostic closure", stats)
	}
}
