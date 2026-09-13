package omnivoice

import (
	"encoding/json"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"os"
	"path/filepath"
	"testing"
)

func TestOpenWeightsPathsAndShards(t *testing.T) {
	for _, sharded := range []bool{false, true} {
		t.Run(map[bool]string{false: "single", true: "sharded"}[sharded], func(t *testing.T) {
			dir := t.TempDir()
			cfg := sampleConfig(t)
			raw, _ := json.Marshal(cfg)
			configPath := filepath.Join(dir, "config.json")
			if err := os.WriteFile(configPath, raw, 0600); err != nil {
				t.Fatal(err)
			}
			specs, _ := expectedTensorSpecs(cfg)
			name := "model.safetensors"
			if sharded {
				name = "part.safetensors"
			}
			writeSyntheticSafetensors(t, filepath.Join(dir, name), specs)
			if sharded {
				wm := map[string]string{}
				for k := range specs {
					wm[k] = name
				}
				data, _ := json.Marshal(map[string]any{"weight_map": wm})
				if err := os.WriteFile(filepath.Join(dir, checkpointIndexName), data, 0600); err != nil {
					t.Fatal(err)
				}
				os.WriteFile(filepath.Join(dir, checkpointFileName), []byte("invalid decoy"), 0600)
			}
			for _, path := range []string{dir, configPath} {
				w, err := OpenWeights(path)
				if err != nil {
					t.Fatal(err)
				}
				layer, err := w.Layer(0)
				if err != nil {
					t.Fatal(err)
				}
				if len(layer) != 11 {
					t.Fatalf("layer has %d tensors", len(layer))
				}
				if _, err = w.Layer(-1); err == nil {
					t.Fatal("negative layer accepted")
				}
				w.Close()
			}
		})
	}
}
func TestFloatingStorageTypes(t *testing.T) {
	cfg := sampleConfig(t)
	specs, _ := expectedTensorSpecs(cfg)
	for _, dtype := range []string{"F16", "F32", "BF16"} {
		infos := map[string]safetensors.TensorInfo{}
		for name, spec := range specs {
			dt := dtype
			if name == "codebook_layer_offsets" {
				dt = "I64"
			}
			infos[name] = safetensors.TensorInfo{DType: dt, Shape: int64sToInts(t, spec.Shape)}
		}
		if meta := ValidateTensorInfos(cfg, infos); !meta.Valid {
			t.Fatalf("%s: %+v", dtype, meta)
		}
	}
}
