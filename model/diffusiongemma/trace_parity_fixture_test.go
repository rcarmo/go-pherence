package diffusiongemma

import (
	"encoding/json"
	"os"
	"testing"
)

type traceMetric struct {
	RMS    float64 `json:"rms"`
	MaxAbs float64 `json:"max_abs"`
}

type traceOpSummary struct {
	Llama traceMetric `json:"llama"`
	Go    traceMetric `json:"go"`
	Delta traceMetric `json:"delta"`
}

type traceLayerSummary struct {
	Layer int                       `json:"layer"`
	Ops   map[string]traceOpSummary `json:"ops"`
}

type phaseAlignedTraceSummary struct {
	Case       string              `json:"case"`
	PromptIDs  []int               `json:"prompt_ids"`
	Row        int                 `json:"row"`
	GoStep     int                 `json:"go_step"`
	EncSeq     int                 `json:"enc_seq"`
	Ops        []string            `json:"ops"`
	Tolerances map[string]float64  `json:"tolerances"`
	Layers     []traceLayerSummary `json:"layers"`
}

func loadPhaseAlignedTraceSummary(t *testing.T) phaseAlignedTraceSummary {
	t.Helper()
	b, err := os.ReadFile("testdata/gguf_hi_phase_aligned_row28_trace_summary.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture phaseAlignedTraceSummary
	if err := json.Unmarshal(b, &fixture); err != nil {
		t.Fatal(err)
	}
	return fixture
}

func TestGGUFHiPhaseAlignedTraceStructuralParityGate(t *testing.T) {
	fixture := loadPhaseAlignedTraceSummary(t)
	if fixture.Case != "gguf_hi_phase_aligned_row28_decode" || fixture.Row != 28 || fixture.GoStep != 48 || fixture.EncSeq != 17 {
		t.Fatalf("unexpected phase-aligned trace metadata: %+v", fixture)
	}
	if len(fixture.PromptIDs) != 17 || fixture.PromptIDs[0] != 2 || fixture.PromptIDs[len(fixture.PromptIDs)-1] != 107 {
		t.Fatalf("unexpected prompt ids: %v", fixture.PromptIDs)
	}
	rmsTol := fixture.Tolerances["structural_rms"]
	maxTol := fixture.Tolerances["structural_max_abs"]
	if rmsTol <= 0 || maxTol <= 0 {
		t.Fatalf("missing structural tolerances: %+v", fixture.Tolerances)
	}
	for _, layer := range fixture.Layers {
		for _, op := range []string{"ffn_post_norm", "l_out"} {
			s, ok := layer.Ops[op]
			if !ok {
				t.Fatalf("layer %d missing op %s", layer.Layer, op)
			}
			if s.Delta.RMS > rmsTol || s.Delta.MaxAbs > maxTol {
				t.Fatalf("layer %d %s delta rms=%g max=%g exceeds structural tolerances rms=%g max=%g (llama=%+v go=%+v)", layer.Layer, op, s.Delta.RMS, s.Delta.MaxAbs, rmsTol, maxTol, s.Llama, s.Go)
			}
		}
	}
}

func TestGGUFHiPhaseAlignedTraceBranchDriftIsBounded(t *testing.T) {
	fixture := loadPhaseAlignedTraceSummary(t)
	rmsTol := fixture.Tolerances["branch_rms"]
	maxTol := fixture.Tolerances["branch_max_abs"]
	if rmsTol <= 0 || maxTol <= 0 {
		t.Fatalf("missing branch tolerances: %+v", fixture.Tolerances)
	}
	for _, layer := range fixture.Layers {
		for _, op := range []string{"attn_out", "ffn_mlp", "ffn_moe", "ffn_moe_combined"} {
			s, ok := layer.Ops[op]
			if !ok {
				t.Fatalf("layer %d missing op %s", layer.Layer, op)
			}
			if s.Delta.RMS > rmsTol || s.Delta.MaxAbs > maxTol {
				t.Fatalf("layer %d %s branch delta rms=%g max=%g exceeds tolerances rms=%g max=%g (llama=%+v go=%+v)", layer.Layer, op, s.Delta.RMS, s.Delta.MaxAbs, rmsTol, maxTol, s.Llama, s.Go)
			}
		}
	}
}
