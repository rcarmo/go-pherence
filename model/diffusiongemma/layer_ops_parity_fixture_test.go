package diffusiongemma

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

type opTraceFixtureMetric struct {
	RMS    float64   `json:"rms"`
	MaxAbs float64   `json:"max_abs"`
	MaxIdx int       `json:"max_idx"`
	First4 []float64 `json:"first4"`
}

type opTraceSummary struct {
	Llama opTraceFixtureMetric `json:"llama"`
	Go    opTraceFixtureMetric `json:"go"`
	Delta struct {
		RMS    float64 `json:"rms"`
		MaxAbs float64 `json:"max_abs"`
	} `json:"delta"`
}

type layerOpsTraceFixture struct {
	Case       string                    `json:"case"`
	PromptIDs  []int                     `json:"prompt_ids"`
	Row        int                       `json:"row"`
	GoStep     int                       `json:"go_step"`
	EncSeq     int                       `json:"enc_seq"`
	Layer      int                       `json:"layer,omitempty"`
	Ops        map[string]opTraceSummary `json:"ops"`
	Tolerances map[string]float64        `json:"tolerances"`
}

func TestGGUFHiPhaseAlignedLayerOpsParityGate(t *testing.T) {
	cases := []struct {
		path  string
		name  string
		layer int
	}{
		{path: "testdata/gguf_hi_phase_aligned_row28_layer0_ops.json", name: "gguf_hi_phase_aligned_row28_layer0_ops", layer: 0},
		{path: "testdata/gguf_hi_phase_aligned_row28_layer1_ops.json", name: "gguf_hi_phase_aligned_row28_layer1_ops", layer: 1},
		{path: "testdata/gguf_hi_phase_aligned_row28_layer2_ops.json", name: "gguf_hi_phase_aligned_row28_layer2_ops", layer: 2},
		{path: "testdata/gguf_hi_phase_aligned_row28_layer3_ops.json", name: "gguf_hi_phase_aligned_row28_layer3_ops", layer: 3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			data, err := os.ReadFile(tc.path)
			if err != nil {
				t.Fatal(err)
			}
			var fixture layerOpsTraceFixture
			if err := json.Unmarshal(data, &fixture); err != nil {
				t.Fatal(err)
			}
			if fixture.Case != tc.name || fixture.Row != 28 || fixture.GoStep != 48 || fixture.EncSeq != 17 {
				t.Fatalf("unexpected layer op fixture metadata: %+v", fixture)
			}
			if fixture.Layer != 0 && fixture.Layer != tc.layer {
				t.Fatalf("fixture layer=%d want %d", fixture.Layer, tc.layer)
			}
			if len(fixture.PromptIDs) != 17 || fixture.PromptIDs[0] != 2 || fixture.PromptIDs[11] != 2202 {
				t.Fatalf("unexpected prompt IDs: %v", fixture.PromptIDs)
			}
			rmsTol := fixture.Tolerances["rms"]
			maxTol := fixture.Tolerances["max_abs"]
			firstTol := fixture.Tolerances["first4"]
			for _, op := range []string{"attn_norm", "attn_out", "ffn_mlp", "ffn_moe", "ffn_moe_combined", "ffn_post_norm", "l_out"} {
				s, ok := fixture.Ops[op]
				if !ok {
					t.Fatalf("missing op %s", op)
				}
				if s.Delta.RMS > rmsTol || s.Delta.MaxAbs > maxTol {
					t.Fatalf("%s delta rms=%g max=%g exceeds tolerances rms=%g max=%g (llama=%+v go=%+v)", op, s.Delta.RMS, s.Delta.MaxAbs, rmsTol, maxTol, s.Llama, s.Go)
				}
				if s.Llama.MaxIdx != s.Go.MaxIdx && (op == "attn_norm" || op == "ffn_post_norm" || op == "l_out") {
					t.Fatalf("%s max_idx mismatch llama=%d go=%d", op, s.Llama.MaxIdx, s.Go.MaxIdx)
				}
				if op == "attn_norm" || op == "l_out" {
					for i := range s.Llama.First4 {
						if d := math.Abs(s.Go.First4[i] - s.Llama.First4[i]); d > firstTol {
							t.Fatalf("%s first4[%d] delta=%g exceeds tolerance: llama=%v go=%v", op, i, d, s.Llama.First4, s.Go.First4)
						}
					}
				}
			}
		})
	}
}
