package pockettts

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestPocketLinearLoadRejectsMalformedWeightsAndBias(t *testing.T) {
	for _, name := range []string{"weight_nan", "weight_inf", "bias_nan", "bias_inf", "bias_shape", "bias_missing", "unsupported_dtype"} {
		t.Run(name, func(t *testing.T) {
			weight := []float32{1, 2, 3, 4}
			bias := []float32{.1, .2}
			dtype := "F32"
			biasShape := []int{2}
			switch name {
			case "weight_nan":
				weight[0] = float32(math.NaN())
			case "weight_inf":
				weight[0] = float32(math.Inf(1))
			case "bias_nan":
				bias[0] = float32(math.NaN())
			case "bias_inf":
				bias[0] = float32(math.Inf(-1))
			case "bias_shape":
				biasShape = []int{1, 2}
			case "unsupported_dtype":
				dtype = "I32"
			}
			raw := make([]byte, 4*(len(weight)+len(bias)))
			for i, v := range append(weight, bias...) {
				binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
			}
			entries := map[string]any{"linear.weight": map[string]any{"dtype": dtype, "shape": []int{2, 2}, "data_offsets": []int{0, 16}}}
			if name != "bias_missing" {
				entries["linear.bias"] = map[string]any{"dtype": "F32", "shape": biasShape, "data_offsets": []int{16, 24}}
			} else {
				raw = raw[:16]
			}
			header, err := json.Marshal(entries)
			if err != nil {
				t.Fatal(err)
			}
			data := make([]byte, 8)
			binary.LittleEndian.PutUint64(data, uint64(len(header)))
			data = append(append(data, header...), raw...)
			path := filepath.Join(t.TempDir(), "malformed.safetensors")
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
			src, err := safetensors.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer src.Close()
			if _, err := loadLinearF32(src, "linear", 2, 2, true); err == nil {
				t.Fatal("invalid tensor accepted")
			}
		})
	}
}
