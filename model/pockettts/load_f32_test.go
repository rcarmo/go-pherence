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

func linearLoadFixture(t *testing.T, dtype string, values []float32, shape []int) *safetensors.File {
	t.Helper()
	var raw []byte
	if dtype == "BF16" {
		raw = make([]byte, 2*len(values))
		for i, v := range values {
			binary.LittleEndian.PutUint16(raw[i*2:], uint16(math.Float32bits(v)>>16))
		}
	} else if dtype == "F16" {
		raw = []byte{0, 0x3c, 0, 0x40, 0, 0x42, 0, 0x44}
	}
	weight := pocketExportTensor{name: "linear.weight", dtype: dtype, shape: shape, values: values}
	if raw != nil {
		weight.values = nil
		weight.raw = raw
	}
	path := filepath.Join(t.TempDir(), "linear.safetensors")
	if dtype == "F16" {
		// The production exporter intentionally emits only F32/BF16. Build
		// this legacy loader input separately rather than broaden export.
		header, err := json.Marshal(map[string]any{"linear.weight": map[string]any{
			"dtype": "F16", "shape": shape, "data_offsets": []int{0, len(raw)},
		}})
		if err != nil {
			t.Fatal(err)
		}
		data := make([]byte, 8)
		binary.LittleEndian.PutUint64(data, uint64(len(header)))
		data = append(append(data, header...), raw...)
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
	} else if err := writePocketExport(path, []pocketExportTensor{{name: "linear.bias", dtype: "F32", shape: []int{2}, values: []float32{.1, -.2}}, weight}); err != nil {
		t.Fatal(err)
	}
	src, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := src.Close(); err != nil {
			t.Error(err)
		}
	})
	return src
}

func TestPocketF32LinearLoadPreservesExportPrecision(t *testing.T) {
	values := []float32{1.001953125, -.998046875, .1000123, -.00000112345}
	src := linearLoadFixture(t, "F32", values, []int{2, 2})
	linear, err := loadLinearF32(src, "linear", 2, 2, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(linear.WeightBF16) != 0 || len(linear.Weight) != len(values) {
		t.Fatal("F32 export was implicitly quantized to BF16")
	}
	// Exact tensor persistence contract, not generated-signal acceptance.
	for i, v := range values {
		if linear.Weight[i] != v {
			t.Fatal("F32 tensor value lost", i, v, linear.Weight[i])
		}
	}
	got := make([]float32, 2)
	if err := linear.Forward(got, []float32{.25, -.5}); err != nil {
		t.Fatal(err)
	}
	for row := 0; row < 2; row++ {
		want := float64(values[row*2])*.25 - float64(values[row*2+1])*.5 + float64(linear.Bias[row])
		if math.Abs(float64(got[row])-want) > 8e-8 {
			t.Fatal("independent affine error", row, got[row], want)
		}
	}
}

func TestPocketBF16LinearLoadRetainsStorage(t *testing.T) {
	src := linearLoadFixture(t, "BF16", []float32{1, -2, .5, .125}, []int{2, 2})
	linear, err := loadLinearF32(src, "linear", 2, 2, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(linear.Weight) != 0 || len(linear.WeightBF16) != 4 || linear.Bias != nil {
		t.Fatal("BF16 dispatch/storage changed")
	}
	got := make([]float32, 2)
	if err := linear.Forward(got, []float32{.25, -.5}); err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "BF16 affine", got, []float32{1.25, .0625}, 0)
}

func TestPocketLinearLoadAdmission(t *testing.T) {
	src := linearLoadFixture(t, "F32", []float32{1, 2, 3, 4}, []int{2, 2})
	for _, dims := range [][2]int{{0, 2}, {2, 0}, {1, 4}, {4, 1}} {
		if _, err := loadLinearF32(src, "linear", dims[0], dims[1], false); err == nil {
			t.Fatal("invalid geometry", dims)
		}
	}
	if _, err := loadLinearF32(nil, "linear", 2, 2, false); err == nil {
		t.Fatal("nil source")
	}
	if _, err := loadLinearF32(src, "missing", 2, 2, false); err == nil {
		t.Fatal("missing tensor")
	}
	if _, err := loadLinearF32(src, "linear", int(^uint(0)>>1), 2, false); err == nil {
		t.Fatal("overflow geometry")
	}
}

func TestPocketF16LinearLoadRetainsConversion(t *testing.T) {
	// F16 historically converts through GetBF16; the F32 export fix must not
	// change that existing input contract.
	src := linearLoadFixture(t, "F16", nil, []int{2, 2})
	linear, err := loadLinearF32(src, "linear", 2, 2, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(linear.Weight) != 0 || len(linear.WeightBF16) != 4 {
		t.Fatal("F16 conversion path changed")
	}
	got := make([]float32, 2)
	if err := linear.Forward(got, []float32{.25, -.5}); err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "F16 affine", got, []float32{-.75, -1.25}, 0)
}
