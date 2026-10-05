//go:build linux

package pockettts

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	train "github.com/rcarmo/go-pherence/backends/training"
	"github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func nativeConditionerForwardGraph(lm *FlowLMTrainingCPU, b FlowLMTrainingBatch) (train.Tape, int64, int64, error) {
	rows := 1 + b.VoiceFrames + len(b.TextTokens) + b.Frames
	g, e := lm.buildBackboneTrainingTape(b, make([]float32, rows*lm.Hidden), 2<<30, true)
	if e != nil {
		return train.Tape{}, 0, 0, e
	}
	if e = train.ValidateResidentTapeBindings(g.Tape, g.ResidentBindings, 2<<30); e != nil {
		return train.Tape{}, 0, 0, e
	}
	var source int64
	for _, s := range g.Tape.Tensors {
		n, e := s.Elements()
		if e != nil {
			return train.Tape{}, 0, 0, e
		}
		source += int64(n) * 4
	}
	roots := append([]int(nil), g.Tape.Outputs[:2]...)
	forward, _, e := train.PruneTape(g.Tape, roots, 2<<30)
	if e != nil {
		return train.Tape{}, 0, 0, e
	}
	aliases := map[int]bool{}
	for _, c := range forward.Commands {
		if c.Op == "reshape" {
			aliases[c.Output] = true
		}
		if strings.Contains(c.Op, "backward") {
			return train.Tape{}, 0, 0, fmt.Errorf("reverse survived")
		}
	}
	var payload int64
	for id, s := range forward.Tensors {
		if aliases[id] {
			continue
		}
		n, e := s.Elements()
		if e != nil {
			return train.Tape{}, 0, 0, e
		}
		payload += int64(n) * 4
	}
	return forward, source, payload, nil
}
func nativeConditionerInputs(t *testing.T) (*FlowLMTrainingCPU, FlowLMTrainingBatch) {
	t.Helper()
	var corpus struct{ Rows []PreparedTrainingRow }
	if e := flowCapturePinned(os.Getenv("GO_PHERENCE_POCKET_SHARED_CORPUS"), "004605224b39ee9637ceb30719c76d4d3c8650142d8df8746938bc5b96e80bb9", 1<<20, &corpus); e != nil {
		t.Fatal(e)
	}
	if len(corpus.Rows) != 1 || corpus.Rows[0].Batch.Frames != 14 || corpus.Rows[0].Batch.VoiceFrames != 13 || corpus.Rows[0].ValidFrames != 13 || len(corpus.Rows[0].Batch.TextTokens) != 1 {
		t.Fatal("conditioner row")
	}
	path := releasedAsset(t, "GO_PHERENCE_POCKETTTS_VOICE_MODEL", "fb0dc01b0d4d2e1c905b7a3e0676e3d9c96d5ae460e24e3ab94981805babf997", 219029196)
	f, e := safetensors.Open(path)
	if e != nil {
		t.Fatal(e)
	}
	lm, e := LoadFlowLMTrainingCPU(f, releasedConfig(t))
	ce := f.Close()
	if e != nil || ce != nil {
		t.Fatal(e, ce)
	}
	return lm, corpus.Rows[0].Batch
}
func TestPocketNativeConditionerGraphTiny(t *testing.T) {
	lm := tinyFlowLMTraining()
	b := FlowLMTrainingBatch{Frames: 3, VoiceFrames: 2, TextTokens: []uint32{1}, NormalizedLatents: make([]float32, 3*lm.LatentDim), VoiceLatents: make([]float32, 2*lm.LatentDim)}
	g, s, n, e := nativeConditionerForwardGraph(lm, b)
	if e != nil || s < n || n == 0 {
		t.Fatal(s, n, e)
	}
	values := interpretFlowTape(t, g)
	sequence, _, e := trunkCaptureSequence(lm, b)
	if e != nil {
		t.Fatal(e)
	}
	_, out := lm.Transformer.forwardTraining(len(sequence)/lm.Hidden, sequence)
	assertSliceClose(t, "conditioner sequence tiny", values[0], sequence, 4e-6)
	assertSliceClose(t, "conditioner trunk tiny", values[1], out, 4e-6)
	bad := b
	bad.TextTokens = []uint32{uint32(lm.Vocabulary)}
	if _, _, _, e := nativeConditionerForwardGraph(lm, bad); e == nil {
		t.Fatal("bad token")
	}
}
func TestPocketNativeConditionerPreparation(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_NATIVE_CONDITIONER_PREPARE") != "1" {
		t.Skip("new conditioner source admission")
	}
	lm, b := nativeConditionerInputs(t)
	g, s, n, e := nativeConditionerForwardGraph(lm, b)
	if e != nil {
		t.Fatal(e)
	}
	t.Logf("source=%d forwardPayload=%d tensors=%d commands=%d", s, n, len(g.Tensors), len(g.Commands))
	if e = flowCaptureWrite(filepath.Join(os.Getenv("GO_PHERENCE_POCKET_NATIVE_CONDITIONER_OUTPUT"), "preparation.json"), struct {
		Source, Payload   int64
		Tensors, Commands int
	}{s, n, len(g.Tensors), len(g.Commands)}); e != nil {
		t.Fatal(e)
	}
}
func TestPocketNativeConditionerForwardCapture(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_NATIVE_CONDITIONER_CAPTURE") != "1" {
		t.Skip("explicit composed conditioner native forward")
	}
	lm, b := nativeConditionerInputs(t)
	g, source, payload, e := nativeConditionerForwardGraph(lm, b)
	if e != nil {
		t.Fatal(e)
	}
	sequence, prefix, e := trunkCaptureSequence(lm, b)
	if e != nil {
		t.Fatal(e)
	}
	_, cpuOutput := lm.Transformer.forwardTraining(len(sequence)/lm.Hidden, sequence)
	if os.Getenv("GO_PHERENCE_DISABLE_VULKAN") != "" || os.Getenv("VK_DRIVER_FILES") != "/usr/share/vulkan/icd.d/nvidia_icd.json" {
		t.Fatal("explicit Vulkan")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	if !vulkan.VulkanInit() || vulkan.VulkanDeviceName() != "NVIDIA GeForce RTX 3060" {
		t.Fatal("runtime mismatch no fallback")
	}
	if e = vulkan.VulkanSetMemoryBudget(vulkan.VulkanMemoryBudget{MaxBytes: 2 << 30, MaxAllocations: 4096}); e != nil {
		t.Fatal(e)
	}
	op, e := vulkan.NewVkTrainingF32(ctx)
	if e != nil {
		t.Fatal(e)
	}
	if rss, e := flowCaptureRSS(); e != nil || rss > 6<<30 {
		t.Fatal("host cap", rss, e)
	}
	tape, e := vulkan.NewVkTrainingTape(ctx, op, g, 2<<30)
	if e != nil {
		t.Fatal("construct failure no retry", e)
	}
	usage := vulkan.VulkanMemoryStats()
	rss, e := flowCaptureRSS()
	if e != nil || rss > 6<<30 {
		t.Fatal("RSS cap", rss, e)
	}
	values, e := tape.Run(ctx)
	if e != nil {
		t.Fatal("nativefailure retained no retry", e)
	}
	if len(values) != 2 {
		t.Fatal("outputs")
	}
	report := struct {
		Scope, Device                                                       string
		Source, Payload                                                     int64
		Prefix, Frames, Hidden                                              int
		Sequence, Output, Conditions, CPUSequence, CPUOutput, CPUConditions []float32
		SequenceError, OutputError, ConditionError                          reverseError
		Native, Final                                                       vulkan.VulkanMemoryUsage
		RSS                                                                 uint64
	}{Scope: "Actual native conditioner+transformer forward all14conditions; no reverse/loss/update. CPUreferencescomparison-only, diagnostic4e-6counts notfullstepacceptance.", Device: vulkan.VulkanDeviceName(), Source: source, Payload: payload, Prefix: prefix, Frames: b.Frames, Hidden: lm.Hidden, Sequence: values[0], Output: values[1], Conditions: append([]float32(nil), values[1][prefix*lm.Hidden:]...), CPUSequence: sequence, CPUOutput: cpuOutput, CPUConditions: append([]float32(nil), cpuOutput[prefix*lm.Hidden:]...), Native: usage, RSS: rss}
	report.SequenceError, e = reverseCompare(values[0], sequence)
	if e != nil {
		t.Fatal(e)
	}
	report.OutputError, e = reverseCompare(values[1], cpuOutput)
	if e != nil {
		t.Fatal(e)
	}
	report.ConditionError, e = reverseCompare(report.Conditions, report.CPUConditions)
	if e != nil {
		t.Fatal(e)
	}
	if e = tape.Close(); e != nil {
		t.Fatal("closefailure no retry", e)
	}
	if e = op.Close(); e != nil {
		t.Fatal(e)
	}
	report.Final = vulkan.VulkanMemoryStats()
	if report.Final.Bytes != 0 || report.Final.Allocations != 0 || report.Final.InFlight || report.Final.Uncertain || report.Final.DeviceLost {
		t.Fatal("closure", report.Final)
	}
	if e = flowCaptureWrite(filepath.Join(os.Getenv("GO_PHERENCE_POCKET_NATIVE_CONDITIONER_OUTPUT"), "native-conditioner.json"), report); e != nil {
		t.Fatal(e)
	}
	t.Logf("sequence max=%g output max=%g conditions max=%g diagnosticover4e-6=%d native=%d RSS=%d", report.SequenceError.Max, report.OutputError.Max, report.ConditionError.Max, report.ConditionError.Over4e6, usage.Bytes, rss)
	runtime.KeepAlive(lm)
}
