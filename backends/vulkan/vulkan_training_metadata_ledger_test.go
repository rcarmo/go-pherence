package vulkan

import (
	"encoding/json"
	"os"
	"testing"
	"unsafe"
)

// Pure Go metadata and shader-interface inventory. Never initialises Vulkan,
// calls an API entrypoint, allocates VkDeviceMemory or probes a device.
func TestVulkanOfflineTrainingMetadataLedger(t *testing.T) {
	type shaderRow struct{ Bindings, PushBytes int }
	report := struct {
		Scope   string
		Objects map[string]uint64
		Shaders map[string]shaderRow
	}{
		Scope:   "Go sizeof and shipped shader interfaces only; excludes allocator rounding and opaque Vulkan driver memory.",
		Objects: map[string]uint64{"VkBuf": uint64(unsafe.Sizeof(VkBuf{})), "VkTensorArena": uint64(unsafe.Sizeof(VkTensorArena{})), "arenaState": uint64(unsafe.Sizeof(vkTensorArenaState{})), "VkTensorF32": uint64(unsafe.Sizeof(VkTensorF32{})), "VkTrainingTape": uint64(unsafe.Sizeof(VkTrainingTape{})), "VkF32Stage": uint64(unsafe.Sizeof(VkF32Stage{})), "VkF32Plan": uint64(unsafe.Sizeof(VkF32Plan{})), "planState": uint64(unsafe.Sizeof(vkF32PlanState{})), "planStage": uint64(unsafe.Sizeof(vkF32PlanStage{})), "bufferBinding": uint64(unsafe.Sizeof(vkBufferBinding{})), "pointer": uint64(unsafe.Sizeof(uintptr(0)))},
		Shaders: map[string]shaderRow{},
	}
	for _, row := range []struct {
		name   string
		source []byte
	}{{"linear-forward", spirvTrainingLinearDualForward}, {"linear-backward", spirvTrainingLinearDualBackward}, {"silu-forward", spirvTrainingSiLUDualForward}, {"silu-backward", spirvTrainingSiLUDualAdjoint}, {"norm-forward", spirvTrainingNormDualForward}, {"norm-backward", spirvTrainingNormDualBackward}, {"mod-forward", spirvTrainingModDualForward}, {"mod-backward", spirvTrainingModDualBackward}, {"attention-forward", spirvTrainingAttentionForward}, {"attention-backward", spirvTrainingAttentionBackward}, {"rope", spirvTrainingRoPE}, {"vector", spirvTrainingVector}, {"time-embedding", spirvTrainingTimeEmbedding}, {"embedding", spirvTrainingEmbedding}, {"eos", spirvTrainingEOS}, {"eos-admission", spirvTrainingEOSAdmission}, {"gelu", spirvTrainingGELU}, {"relu", spirvTrainingReLU}, {"row", spirvTrainingRow}, {"lsd-loss", spirvTrainingLSDLoss}} {
		c, err := InspectVulkanShader(row.source)
		if err != nil {
			t.Fatal(row.name, err)
		}
		bits := c.StorageBindings
		count := 0
		for bits != 0 {
			count += int(bits & 1)
			bits >>= 1
		}
		if count < 1 || count > 16 || c.PushBytes > 128 {
			t.Fatal("shader interface", row.name, c)
		}
		report.Shaders[row.name] = shaderRow{Bindings: count, PushBytes: int(c.PushBytes)}
	}
	if report.Objects["bufferBinding"] != 3*report.Objects["pointer"] || len(report.Shaders) != 20 {
		t.Fatal("metadata inventory")
	}
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if path := os.Getenv("GO_PHERENCE_VULKAN_METADATA_OUTPUT"); path != "" {
		f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if err != nil {
			t.Fatal(err)
		}
		_, we := f.Write(append(data, '\n'))
		ce := f.Close()
		if we != nil || ce != nil {
			t.Fatal(we, ce)
		}
	}
	t.Logf("metadata objects=%v shader-interfaces=%v; no native API calls", report.Objects, report.Shaders)
}
