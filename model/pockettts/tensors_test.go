package pockettts

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func releasedConfig(t testing.TB) Config {
	t.Helper()
	data, err := os.ReadFile("testdata/english-upstream.yaml")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseConfig(data)
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestPocketTensorInventorySynthetic(t *testing.T) {
	cfg := releasedConfig(t)
	infos := make(map[string]safetensors.TensorInfo)
	for name, shape := range expectedTensorShapes(cfg) {
		infos[name] = safetensors.TensorInfo{DType: "BF16", Shape: append([]int(nil), shape...)}
	}
	inv := InspectTensorInfos(cfg, infos)
	if !inv.CheckpointReady() || inv.FlowLayers != 6 || inv.FlowBlocks != 6 || inv.MimiEncoderLayers != 2 || inv.MimiDecoderLayers != 2 {
		t.Fatalf("inventory=%+v", inv)
	}
	delete(infos, "flow_lm.bos_emb")
	inv = InspectTensorInfos(cfg, infos)
	if inv.CheckpointReady() || len(inv.Missing) != 1 {
		t.Fatalf("missing inventory=%+v", inv)
	}
}

func TestReleasedEnglishTensorInventory(t *testing.T) {
	path := releasedModel(t)
	f, err := safetensors.Open(path)
	if err != nil {
		t.Skipf("released Pocket TTS checkpoint unavailable: %v", err)
	}
	defer f.Close()
	inv := InspectTensorInfos(releasedConfig(t), f.TensorInfos())
	if !inv.CheckpointReady() || inv.Total != 214 || inv.DTypes["BF16"] != 214 {
		t.Fatalf("released inventory=%+v", inv)
	}
}
