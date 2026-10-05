//go:build linux

package pockettts

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/backends/vulkan"
)

func TestPocketNativeCorrectedBackboneCapture(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_CORRECTED_BACKBONE_NATIVE") != "1" {
		t.Skip("new corrected objective-seed nativebackbone only")
	}
	in := joinedCapturePrepare(t)
	var saved struct{ TrunkSeed []float32 }
	path, pin := "/workspace/exports/pocket-native-parameters-20261004/corrected-seed-input.json", "c4606c55cd69da7738de8e41f4db1841d5c9a870ab599e5074c5eaaf7e554ffb"
	if p := os.Getenv("GO_PHERENCE_POCKET_BACKBONE_SEED_INPUT"); p != "" {
		path, pin = p, os.Getenv("GO_PHERENCE_POCKET_BACKBONE_SEED_SHA256")
	}
	if e := flowCapturePinned(path, pin, 1<<20, &saved); e != nil {
		t.Fatal(e)
	}
	if len(saved.TrunkSeed) != 29696 || !finiteF32(saved.TrunkSeed) || !zeroTrainingSeed(saved.TrunkSeed[:15360]) {
		t.Fatal("corrected seed geometry")
	}
	p, source, payload, e := joinedCaptureBuild(in.LM, in.Batch, saved.TrunkSeed)
	if e != nil {
		t.Fatal(e)
	}
	out, reference, inputs, e := joinedCaptureCPU(in.LM, in.Batch, saved.TrunkSeed)
	if e != nil {
		t.Fatal(e)
	}
	if os.Getenv("GO_PHERENCE_DISABLE_VULKAN") != "" || os.Getenv("VK_DRIVER_FILES") != "/usr/share/vulkan/icd.d/nvidia_icd.json" {
		t.Fatal("explicit Vulkan")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
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
		t.Fatal("hostcap", rss, e)
	}
	owner, e := vulkan.NewVkTrainingTape(ctx, op, p.Tape, 2<<30)
	if e != nil {
		t.Fatal("constructionfailure no retry", e)
	}
	usage := vulkan.VulkanMemoryStats()
	rss, e := flowCaptureRSS()
	if e != nil || rss > 6<<30 {
		t.Fatal("RSScap", rss, e)
	}
	values, e := owner.Run(ctx)
	if e != nil {
		t.Fatal("nativefailure retained no retry", e)
	}
	names := make([]string, 0, len(p.ResidentBindings))
	for name := range p.ResidentBindings {
		names = append(names, name)
	}
	sort.Strings(names)
	native := map[string][]float32{}
	for i, name := range names {
		native[name] = values[4+i]
	}
	root := os.Getenv("GO_PHERENCE_POCKET_PARAMETER_OUTPUT")
	manifest, errDist, e := reverseWriteGradients(filepath.Join(root, "backbone-native.f32"), native, reference)
	if e != nil {
		t.Fatal(e)
	}
	refManifest, _, e := reverseWriteGradients(filepath.Join(root, "backbone-reference.f32"), reference, reference)
	if e != nil {
		t.Fatal(e)
	}
	report := struct {
		Scope, NativeFile, ReferenceFile string
		Source, Payload                  int64
		Manifest, ReferenceManifest      []reverseManifest
		Error, OutputError, TargetError  reverseError
		Native, Final                    vulkan.VulkanMemoryUsage
		RSS                              uint64
	}{Scope: "Correctproductionobjective-seed completebackboneVJP;nooptimizer/globalclip/update/fullstepacceptance.", NativeFile: "backbone-native.f32", ReferenceFile: "backbone-reference.f32", Source: source, Payload: payload, Manifest: manifest, ReferenceManifest: refManifest, Error: errDist, Native: usage, RSS: rss}
	report.OutputError, e = reverseCompare(values[1][15360:], out.Z)
	if e != nil {
		t.Fatal(e)
	}
	report.TargetError, e = reverseCompare(values[3], inputs.NormalizedLatents)
	if e != nil {
		t.Fatal(e)
	}
	if e = owner.Close(); e != nil {
		t.Fatal("closefailure no retry", e)
	}
	if e = op.Close(); e != nil {
		t.Fatal(e)
	}
	report.Final = vulkan.VulkanMemoryStats()
	if report.Final.Bytes != 0 || report.Final.Allocations != 0 || report.Final.InFlight || report.Final.Uncertain || report.Final.DeviceLost {
		t.Fatal("closure", report.Final)
	}
	if e = flowCaptureWrite(filepath.Join(root, "corrected-backbone.json"), report); e != nil {
		t.Fatal(e)
	}
	t.Logf("corrected fullbackbonevalues=%d max=%g RMS=%g diagnosticAbove4e-6=%d native=%d RSS=%d", errDist.Count, errDist.Max, errDist.RMS, errDist.Over4e6, usage.Bytes, rss)
	runtime.KeepAlive(in)
}
