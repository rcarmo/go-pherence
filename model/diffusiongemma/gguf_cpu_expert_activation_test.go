package diffusiongemma

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/internal/ggmlfp16"
)

func TestGGUFExpertGGMLGELUFP16LookupThresholds(t *testing.T) {
	if got := ggmlfp16.GELUFP16Lookup(-10); got != 0 {
		t.Fatalf("lookup(-10)=%v want 0", got)
	}
	if got := ggmlfp16.GELUFP16Lookup(10); got != 10 {
		t.Fatalf("lookup(10)=%v want 10", got)
	}
	if got := ggmlfp16.GELUFP16Lookup(-1); got == 0 {
		t.Fatalf("lookup(-1)=%v want non-zero table value", got)
	}
}

func TestGGUFExpertGGMLGELUFP16MulSynthetic(t *testing.T) {
	gate := []float32{-11, -10, -1.5, 0, 0.5, 9.5, 10, 11}
	up := []float32{3, 2, -4, 5, -6, 7, 8, 9}
	dst := make([]float32, len(gate))
	if !ggufExpertGGMLGELUFP16MulTo(dst, gate, up) {
		t.Fatal("activation rejected valid buffers")
	}
	for i := range dst {
		want := ggmlfp16.GELUFP16Lookup(gate[i]) * up[i]
		if diff := math.Abs(float64(dst[i] - want)); diff > 1e-7 {
			t.Fatalf("dst[%d]=%.9g want=%.9g diff=%.9g", i, dst[i], want, diff)
		}
	}
	if ggufExpertGGMLGELUFP16MulTo(make([]float32, 1), gate, up) {
		t.Fatal("activation accepted mismatched dst")
	}
	if ggufExpertGGMLGELUFP16MulTo(make([]float32, len(gate)), gate[:len(gate)-1], up) {
		t.Fatal("activation accepted mismatched gate")
	}
}

func TestGGUFExpertGGMLGELUFP16MatchesMoeMicrographOracleBoundary(t *testing.T) {
	fixture := struct {
		Dimensions struct {
			Intermediate int `json:"intermediate"`
		} `json:"dimensions"`
		Outputs struct {
			GateUp []float32 `json:"ffn_moe_gate_up"`
			GELU   []float32 `json:"ffn_moe_gelu"`
			Act    []float32 `json:"ffn_moe_act"`
		} `json:"outputs"`
	}{}
	path := filepath.Join("..", "..", "loader", "gguf", "testdata", "actual_ggml_moe_micrograph_oracle.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read oracle fixture: %v", err)
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatalf("decode oracle fixture: %v", err)
	}
	intermediate := fixture.Dimensions.Intermediate
	if intermediate <= 0 || len(fixture.Outputs.GateUp) == 0 || len(fixture.Outputs.GELU) == 0 || len(fixture.Outputs.GELU) != len(fixture.Outputs.Act) || len(fixture.Outputs.GateUp) != 2*len(fixture.Outputs.GELU) || len(fixture.Outputs.GELU)%intermediate != 0 {
		t.Fatalf("unexpected oracle sizes gate_up=%d gelu=%d act=%d intermediate=%d", len(fixture.Outputs.GateUp), len(fixture.Outputs.GELU), len(fixture.Outputs.Act), intermediate)
	}
	gate := make([]float32, len(fixture.Outputs.GELU))
	up := make([]float32, len(fixture.Outputs.GELU))
	for base := 0; base < len(fixture.Outputs.GELU); base += intermediate {
		guBase := (base / intermediate) * (2 * intermediate)
		copy(gate[base:base+intermediate], fixture.Outputs.GateUp[guBase:guBase+intermediate])
		copy(up[base:base+intermediate], fixture.Outputs.GateUp[guBase+intermediate:guBase+2*intermediate])
	}
	gotGELU := make([]float32, len(gate))
	for i, gateValue := range gate {
		gotGELU[i] = ggmlfp16.GELUFP16Lookup(gateValue)
		if diff := math.Abs(float64(gotGELU[i] - fixture.Outputs.GELU[i])); diff > 1e-6 {
			t.Fatalf("gelu[%d]=%.9g want=%.9g diff=%.9g", i, gotGELU[i], fixture.Outputs.GELU[i], diff)
		}
	}
	gotAct := make([]float32, len(gate))
	if !ggufExpertGGMLGELUFP16MulTo(gotAct, gate, up) {
		t.Fatal("activation rejected oracle buffers")
	}
	for i := range gotAct {
		if diff := math.Abs(float64(gotAct[i] - fixture.Outputs.Act[i])); diff > 1e-6 {
			t.Fatalf("act[%d]=%.9g want=%.9g diff=%.9g", i, gotAct[i], fixture.Outputs.Act[i], diff)
		}
	}
}
