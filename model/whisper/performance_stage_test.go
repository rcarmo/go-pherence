package whisper

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/audio/media"
	legacy "github.com/rcarmo/go-pherence/loader/whisperggml"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func whisperStageMode(backend string) (vulkanLinearMode, error) {
	switch backend {
	case "", "vulkan-f32":
		return vulkanDefaultLinearMode, nil
	case "vulkan-f32-tile64-key32-scoreilp":
		return vulkanLinearF32Tile64Key32ScoreILP, nil
	case "vulkan-original-q5-mlp":
		return vulkanLinearOriginalQ5MLP, nil
	case "vulkan-original-q5-integer-dot-fc1":
		return vulkanLinearOriginalQ5IntegerDotFC1, nil
	case "vulkan-original-q5-exact-combined":
		return vulkanLinearOriginalQ5ExactCombined, nil
	case "vulkan-original-q5-attention-outputilp":
		return vulkanLinearOriginalQ5AttentionOutputILP, nil
	case "vulkan-original-q5-padded-integer-dot-all":
		return vulkanLinearOriginalQ5PaddedIntegerDotAll, nil
	case "vulkan-original-q5-padded-integer-dot-mmq":
		return vulkanLinearOriginalQ5PaddedIntegerDotMMQ, nil
	case "vulkan-original-q5-padded-integer-dot-mmq-tanh":
		return vulkanLinearOriginalQ5PaddedIntegerDotMMQTanh, nil
	default:
		return 0, fmt.Errorf("unsupported stage-profile backend %q", backend)
	}
}

func TestWhisperStageModeAdmission(t *testing.T) {
	for backend, want := range map[string]vulkanLinearMode{"": vulkanDefaultLinearMode, "vulkan-f32": vulkanDefaultLinearMode, "vulkan-f32-tile64-key32-scoreilp": vulkanLinearF32Tile64Key32ScoreILP, "vulkan-original-q5-mlp": vulkanLinearOriginalQ5MLP, "vulkan-original-q5-integer-dot-fc1": vulkanLinearOriginalQ5IntegerDotFC1, "vulkan-original-q5-exact-combined": vulkanLinearOriginalQ5ExactCombined, "vulkan-original-q5-attention-outputilp": vulkanLinearOriginalQ5AttentionOutputILP, "vulkan-original-q5-padded-integer-dot-all": vulkanLinearOriginalQ5PaddedIntegerDotAll, "vulkan-original-q5-padded-integer-dot-mmq": vulkanLinearOriginalQ5PaddedIntegerDotMMQ, "vulkan-original-q5-padded-integer-dot-mmq-tanh": vulkanLinearOriginalQ5PaddedIntegerDotMMQTanh} {
		got, err := whisperStageMode(backend)
		if err != nil || got != want {
			t.Fatal(backend, got, err)
		}
	}
	for _, backend := range []string{"cpu", "vulkan-q8-mlp", "vulkan-f32-tile64-key32", "unknown"} {
		if _, err := whisperStageMode(backend); err == nil {
			t.Fatal("unknown backend admitted", backend)
		}
	}
}

// Diagnostic only: single-stage submissions change barriers/submission costs.
// Whole-request benchmark remains the acceptance measurement.
func TestWhisperPerformanceStageProfile(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_WHISPER_STAGES") != "1" {
		t.Skip("explicit native diagnostic window")
	}
	deadline, ok := t.Deadline()
	if !ok || time.Until(deadline) > 10*time.Minute {
		t.Fatal("bounded diagnostic")
	}
	ctx, cancel := context.WithDeadline(context.Background(), deadline.Add(-time.Second))
	defer cancel()
	mode, err := whisperStageMode(os.Getenv("GO_PHERENCE_WHISPER_BENCH_BACKEND"))
	if err != nil {
		t.Fatal(err)
	}
	var model *Whisper
	var file *legacy.File
	if mode == vulkanLinearOriginalQ5MLP || mode == vulkanLinearOriginalQ5IntegerDotFC1 || mode == vulkanLinearOriginalQ5ExactCombined || mode == vulkanLinearOriginalQ5AttentionOutputILP || (mode == vulkanLinearOriginalQ5PaddedIntegerDotAll || mode == vulkanLinearOriginalQ5PaddedIntegerDotMMQ || mode == vulkanLinearOriginalQ5PaddedIntegerDotMMQTanh) {
		model, _, _, file = pinnedLegacyWhisperModelOpen(t, ctx)
		defer file.Close()
	} else {
		model, _, _ = pinnedTurboSpeechModel(t, ctx)
	}
	path := os.Getenv("GO_PHERENCE_WHISPER_BENCH_INPUT")
	pinnedSpeechFile(t, path, os.Getenv("GO_PHERENCE_WHISPER_BENCH_INPUT_SHA256"))
	init := vk.VulkanInit
	if mode == vulkanLinearOriginalQ5IntegerDotFC1 || (mode == vulkanLinearOriginalQ5PaddedIntegerDotAll || mode == vulkanLinearOriginalQ5PaddedIntegerDotMMQ || mode == vulkanLinearOriginalQ5PaddedIntegerDotMMQTanh) {
		init = vk.VulkanInitIntegerDot
	}
	if !init() {
		t.Fatal("Vulkan init")
	}
	device := os.Getenv("GO_PHERENCE_VULKAN_DEVICE")
	if device == "" || !strings.Contains(strings.ToLower(vk.VulkanDeviceName()), strings.ToLower(device)) || strings.Contains(strings.ToLower(vk.VulkanDeviceName()), "llvmpipe") {
		t.Fatal("explicit physical device required")
	}
	layout, err := describeVulkanEncoder(ctx, model.Encoder, model.Config.MaxLength)
	if err != nil {
		t.Fatal(err)
	}
	var stages []vk.VkF32Stage
	enc, err := newVulkanEncoderPackedMode(ctx, model.Encoder, model.Config.MaxLength, func(ctx context.Context, s []vk.VkF32Stage) (*vk.VkF32Plan, error) {
		stages = append(stages, s...)
		return vk.NewVkF32Plan(ctx, s)
	}, mode, file)
	if err != nil {
		if enc != nil {
			enc.Close()
		}
		t.Fatal(err)
	}
	defer func() {
		if err := enc.Close(); err != nil {
			t.Error(err)
		}
	}()
	var labels, outputs []string
	var inputs [][]string
	for _, p := range layout.plans {
		for _, s := range p {
			if (mode == vulkanLinearOriginalQ5PaddedIntegerDotAll || mode == vulkanLinearOriginalQ5PaddedIntegerDotMMQ || mode == vulkanLinearOriginalQ5PaddedIntegerDotMMQTanh) && s.op == "linear" {
				labels = append(labels, "q8-activation")
				outputs = append(outputs, "q8")
				inputs = append(inputs, []string{s.in[0]})
			}
			if mode == vulkanLinearOriginalQ5IntegerDotFC1 && s.op == "linear" && strings.Contains(s.in[1], ".fc1.") {
				labels = append(labels, "q8-activation")
				outputs = append(outputs, "q8fc1")
				inputs = append(inputs, []string{s.in[0]})
			}
			labels = append(labels, s.op)
			outputs = append(outputs, s.out)
			inputs = append(inputs, append([]string(nil), s.in...))
		}
	}
	if len(labels) != len(stages) {
		t.Fatal("stage geometry")
	}
	plans := make([]*vk.VkF32Plan, len(stages))
	defer func() {
		for _, p := range plans {
			if p != nil {
				if err := p.Close(); err != nil {
					t.Error(err)
				}
			}
		}
	}()
	for i, s := range stages {
		plans[i], err = vk.NewVkF32Plan(ctx, []vk.VkF32Stage{s})
		if err != nil {
			t.Fatal(err)
		}
	}
	reader, err := media.OpenCanonicalPCM(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	if reader.Timeline().Samples > 60*16000 {
		t.Fatal("bounded input")
	}
	pcm := make([]float32, model.Config.MaxLength*160)
	_, err = reader.ReadSamplesAt(ctx, pcm, 0)
	if err != nil && err != io.EOF {
		t.Fatal(err)
	}
	mel, _, err := MelFlatFromSamplesCheckedContext(ctx, pcm, model.Config)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := enc.Forward(ctx, mel)
	if err != nil {
		t.Fatal(err)
	}
	rows := make([]map[string]any, 0, len(plans)*3)
	for run := 0; run < 3; run++ {
		if err := enc.s.input.Upload(ctx, mel); err != nil {
			t.Fatal(err)
		}
		for i, p := range plans {
			if dump := os.Getenv("GO_PHERENCE_WHISPER_STAGE_INPUT_DUMP"); dump != "" && run == 0 && len(inputs[i]) > 1 && strings.HasPrefix(inputs[i][1], "layer0.") && (strings.Contains(inputs[i][1], "fc1") || strings.Contains(inputs[i][1], "fc2")) && labels[i] == "linear" {
				name := inputs[i][0]
				tensor := enc.s.tensors[name]
				shape := tensor.Shape()
				elems := 1
				for _, n := range shape {
					elems *= n
				}
				data := make([]float32, elems)
				if err := tensor.Download(ctx, data); err != nil {
					t.Fatal(err)
				}
				bytes := make([]byte, len(data)*4)
				for j, v := range data {
					binary.LittleEndian.PutUint32(bytes[j*4:], math.Float32bits(v))
				}
				file, err := os.OpenFile(filepath.Join(dump, outputs[i]+"-"+name+".f32"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
				if err != nil {
					t.Fatal(err)
				}
				_, err = file.Write(bytes)
				closeErr := file.Close()
				if err != nil || closeErr != nil {
					t.Fatal(err, closeErr)
				}
			}
			start := time.Now()
			if err := p.Run(ctx); err != nil {
				t.Fatal(err)
			}
			rows = append(rows, map[string]any{"run": run, "stage": i, "op": labels[i], "out": outputs[i], "inputs": inputs[i], "backend": os.Getenv("GO_PHERENCE_WHISPER_BENCH_BACKEND"), "seconds": time.Since(start).Seconds()})
		}
		out := make([]float32, len(baseline))
		if err := enc.s.output.Download(ctx, out); err != nil {
			t.Fatal(err)
		}
		for i := range out {
			if out[i] != baseline[i] {
				t.Fatalf("single-stage output changed at %d", i)
			}
		}
	}
	b, err := json.MarshalIndent(rows, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	out := os.Getenv("GO_PHERENCE_WHISPER_STAGES_REPORT")
	if out == "" {
		t.Fatal("explicit report")
	}
	f, err := os.OpenFile(out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		t.Fatal(err)
	}
	_, err = f.Write(b)
	closeErr := f.Close()
	if err != nil || closeErr != nil {
		t.Fatal(err, closeErr)
	}
}
