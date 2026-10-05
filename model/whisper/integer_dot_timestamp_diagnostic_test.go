package whisper

import (
	"context"
	"encoding/json"
	"fmt"
	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/audio/media"
	"io"
	"math"
	"os"
	"sort"
	"strings"
	"testing"
	"time"
)

type intDotTraceScore struct {
	Token       int
	Text        string
	Raw, Masked float32
}
type intDotTraceStep struct {
	Index, Selected int
	Scores          []intDotTraceScore
}
type intDotTimestampTrace struct {
	Mode     string
	Segments []Segment
	Steps    []intDotTraceStep
}

// Test-only recorder. Copies logits after ForwardToken; it never changes decoder
// inputs, masks, state, selected tokens, or final timestamp semantics.
func TestWhisperIntegerDotTimestampDiagnostic(t *testing.T) {
	report := os.Getenv("GO_PHERENCE_TEST_INTDOT_TIMESTAMP_REPORT")
	if report == "" {
		t.Skip("explicit timestamp logit diagnostic")
	}
	deadline, ok := t.Deadline()
	if !ok || time.Until(deadline) > 120*time.Second {
		t.Fatal("bounded120s")
	}
	ctx, cancel := context.WithDeadline(context.Background(), deadline.Add(-time.Second))
	defer cancel()
	model, tokenizer, _, file := pinnedLegacyWhisperModelOpen(t, ctx)
	defer file.Close()
	if !vk.VulkanInitIntegerDot() || os.Getenv("GO_PHERENCE_VULKAN_DEVICE") == "" || !strings.Contains(vk.VulkanDeviceName(), os.Getenv("GO_PHERENCE_VULKAN_DEVICE")) {
		t.Fatal("physical device")
	}
	path := os.Getenv("GO_PHERENCE_WHISPER_BENCH_INPUT")
	pinnedSpeechFile(t, path, os.Getenv("GO_PHERENCE_WHISPER_BENCH_INPUT_SHA256"))
	reader, e := media.OpenCanonicalPCM(ctx, path)
	if e != nil {
		t.Fatal(e)
	}
	defer reader.Close()
	samples := make([]float32, model.Config.MaxLength*160)
	_, e = reader.ReadSamplesAt(ctx, samples, 0)
	if e != nil && e != io.EOF {
		t.Fatal(e)
	}
	mel, _, e := MelFlatFromSamplesCheckedContext(ctx, samples, model.Config)
	if e != nil {
		t.Fatal(e)
	}
	vocab, e := checkedTimestampVocabulary(model.Config, tokenizer, "pt")
	if e != nil {
		t.Fatal(e)
	}
	var traces []intDotTimestampTrace
	modes := []string{"baseline", "integer-dot"}
	if os.Getenv("GO_PHERENCE_TEST_INTDOT_PARTIAL") == "1" {
		modes = []string{"fc1-only", "fc2-only", "early-half", "late-half"}
	}
	for _, mode := range modes {
		var enc *VulkanEncoder
		if mode == "baseline" {
			enc, e = NewVulkanEncoderOriginalQ5MLP(ctx, model.Encoder, model.Config.MaxLength, file)
		} else if mode == "integer-dot" {
			enc, e = NewVulkanEncoderOriginalQ5IntegerDotMLP(ctx, model.Encoder, model.Config.MaxLength, file)
		} else {
			selectWeight := func(name string) bool {
				switch mode {
				case "fc1-only":
					return strings.Contains(name, ".fc1.")
				case "fc2-only":
					return strings.Contains(name, ".fc2.")
				case "early-half", "late-half":
					var layer int
					if _, err := fmt.Sscanf(name, "layer%d.", &layer); err != nil {
						t.Fatal(err)
					}
					return (layer < 16) == (mode == "early-half")
				}
				return false
			}
			enc, e = newVulkanEncoderPackedSelected(ctx, model.Encoder, model.Config.MaxLength, vk.NewVkF32Plan, vulkanLinearOriginalQ5IntegerDotMLP, file, false, selectWeight)
		}
		if e != nil {
			t.Fatal(e)
		}
		hidden, e := enc.Forward(ctx, mel)
		if e != nil {
			t.Fatal(e)
		}
		if e = enc.Close(); e != nil {
			t.Fatal(e)
		}
		state, e := NewDecoderStateContext(ctx, model.Config, hidden, len(hidden)/model.Config.EncoderDModel, model.Decoder)
		if e != nil {
			t.Fatal(e)
		}
		var recorded [][]float32
		var consumed []int
		forward := func(tok int) ([]float32, error) {
			if e := ctx.Err(); e != nil {
				return nil, e
			}
			logits := model.Decoder.ForwardToken(tok, state)
			consumed = append(consumed, tok)
			recorded = append(recorded, append([]float32(nil), logits...))
			return logits, nil
		}
		opts := PCMTranscribeOptions{Language: "pt", MaxInitialTimestampIndex: 50}
		segments, e := decodeCheckedTimestamps(ctx, model.Config, tokenizer, vocab, opts, model.Decoder.SuppressTokens, model.Decoder.BeginSuppressTokens, forward)
		if e != nil {
			t.Fatal(e)
		}
		trace := intDotTimestampTrace{Mode: mode, Segments: segments}
		var generated []int
		for i, raw := range recorded[2:] {
			masked := append([]float32(nil), raw...)
			for _, id := range model.Decoder.SuppressTokens {
				masked[id] = float32(math.Inf(-1))
			}
			if i == 0 {
				for _, id := range model.Decoder.BeginSuppressTokens {
					masked[id] = float32(math.Inf(-1))
				}
			}
			checkedTimestampMask(masked, generated, vocab, 50)
			selected := argmax(masked)
			if i+3 < len(consumed) && selected != consumed[i+3] {
				t.Fatal("trace replay diverged from actual decoder", mode, i)
			}
			ids := make([]int, len(raw))
			for id := range ids {
				ids[id] = id
			}
			sort.Slice(ids, func(i, j int) bool { return masked[ids[i]] > masked[ids[j]] })
			watch := append([]int(nil), ids[:5]...)
			watch = append(watch, vocab.eot, vocab.timestampBegin+368, vocab.timestampBegin+452)
			step := intDotTraceStep{Index: i, Selected: selected}
			seen := map[int]bool{}
			for _, id := range watch {
				if seen[id] {
					continue
				}
				seen[id] = true
				r, m := raw[id], masked[id]
				if math.IsInf(float64(m), -1) {
					m = -1e30
				}
				step.Scores = append(step.Scores, intDotTraceScore{Token: id, Text: tokenizer.Vocab[id], Raw: r, Masked: m})
			}
			trace.Steps = append(trace.Steps, step)
			if selected == vocab.eot {
				break
			}
			generated = append(generated, selected)
		}
		traces = append(traces, trace)
	}
	bytes, e := json.MarshalIndent(traces, "", "  ")
	if e != nil {
		t.Fatal(e)
	}
	f, e := os.OpenFile(report, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		t.Fatal(e)
	}
	_, e = f.Write(bytes)
	ce := f.Close()
	if e != nil || ce != nil {
		t.Fatal(e, ce)
	}
	for _, trace := range traces {
		t.Logf("TIMESTAMP_TRACE mode=%s steps=%d segments=%+v", trace.Mode, len(trace.Steps), trace.Segments)
	}
}
