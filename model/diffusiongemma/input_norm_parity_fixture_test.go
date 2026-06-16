package diffusiongemma

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

type inputNormTraceMetric struct {
	RMS    float64   `json:"rms"`
	MaxAbs float64   `json:"max_abs"`
	MaxIdx int       `json:"max_idx"`
	First4 []float64 `json:"first4"`
}

type inputNormTraceFixture struct {
	Case       string                          `json:"case"`
	PromptIDs  []int                           `json:"prompt_ids"`
	Row        int                             `json:"row"`
	GoStep     int                             `json:"go_step"`
	EncSeq     int                             `json:"enc_seq"`
	Llama      map[string]inputNormTraceMetric `json:"llama"`
	Go         map[string]inputNormTraceMetric `json:"go"`
	Tolerances map[string]float64              `json:"tolerances"`
}

func TestGGUFHiPhaseAlignedInputNormParityGate(t *testing.T) {
	data, err := os.ReadFile("testdata/gguf_hi_phase_aligned_row0_input_norm_summary.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture inputNormTraceFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Case != "gguf_hi_phase_aligned_row0_input_norm" || fixture.Row != 0 || fixture.GoStep != 48 || fixture.EncSeq != 17 {
		t.Fatalf("unexpected input norm fixture metadata: %+v", fixture)
	}
	if len(fixture.PromptIDs) != 17 || fixture.PromptIDs[0] != 2 || fixture.PromptIDs[11] != 2202 {
		t.Fatalf("unexpected prompt IDs: %v", fixture.PromptIDs)
	}
	llamaNorm, ok := fixture.Llama["attn_norm_0"]
	if !ok {
		t.Fatal("missing llama attn_norm_0")
	}
	goNorm, ok := fixture.Go["input_norm_0"]
	if !ok {
		t.Fatal("missing Go input_norm_0")
	}
	if d := math.Abs(goNorm.RMS - llamaNorm.RMS); d > fixture.Tolerances["input_norm_rms"] {
		t.Fatalf("input_norm RMS delta=%g exceeds tolerance: llama=%+v go=%+v", d, llamaNorm, goNorm)
	}
	if d := math.Abs(goNorm.MaxAbs - llamaNorm.MaxAbs); d > fixture.Tolerances["input_norm_max_abs"] {
		t.Fatalf("input_norm max_abs delta=%g exceeds tolerance: llama=%+v go=%+v", d, llamaNorm, goNorm)
	}
	if goNorm.MaxIdx != llamaNorm.MaxIdx {
		t.Fatalf("input_norm max_idx mismatch llama=%d go=%d", llamaNorm.MaxIdx, goNorm.MaxIdx)
	}
	for i := range llamaNorm.First4 {
		if d := math.Abs(goNorm.First4[i] - llamaNorm.First4[i]); d > 1e-5 {
			t.Fatalf("input_norm first4[%d] delta=%g exceeds tolerance: llama=%v go=%v", i, d, llamaNorm.First4, goNorm.First4)
		}
	}
	llamaOut, ok := fixture.Llama["l_out_0"]
	if !ok {
		t.Fatal("missing llama l_out_0")
	}
	goOut, ok := fixture.Go["l_out_0"]
	if !ok {
		t.Fatal("missing Go l_out_0")
	}
	if d := math.Abs(goOut.RMS - llamaOut.RMS); d > fixture.Tolerances["l_out_rms"] {
		t.Fatalf("l_out RMS delta=%g exceeds tolerance: llama=%+v go=%+v", d, llamaOut, goOut)
	}
	if d := math.Abs(goOut.MaxAbs - llamaOut.MaxAbs); d > fixture.Tolerances["l_out_max_abs"] {
		t.Fatalf("l_out max_abs delta=%g exceeds tolerance: llama=%+v go=%+v", d, llamaOut, goOut)
	}
}
