package diffusiongemma

import "testing"

func TestExecutionGraphMatchesLlamaPhaseRegionSemantics(t *testing.T) {
	unified := BuildExecutionGraph(ExecutionGraphUnified, 9, 256)
	if unified.Phase != ExecutionGraphUnified || unified.PromptLength != 9 || unified.CanvasLength != 256 || unified.TokenCount != 265 {
		t.Fatalf("unified graph = %+v", unified)
	}
	if got := unified.CanvasPosition(0); got != 9 {
		t.Fatalf("unified canvas position 0 = %d, want 9", got)
	}

	prefill := BuildExecutionGraph(ExecutionGraphPrefill, 9, 256)
	if prefill.Phase != ExecutionGraphPrefill || prefill.PromptLength != 9 || prefill.CanvasLength != 0 || prefill.TokenCount != 9 {
		t.Fatalf("prefill graph = %+v", prefill)
	}

	decode := BuildExecutionGraph(ExecutionGraphDecode, 9, 256)
	if decode.Phase != ExecutionGraphDecode || decode.PromptLength != 9 || decode.CanvasLength != 256 || decode.TokenCount != 256 {
		t.Fatalf("decode graph = %+v", decode)
	}
	if got := decode.CanvasPosition(0); got != 9 {
		t.Fatalf("decode canvas position 0 = %d, want 9", got)
	}
	if got := decode.CanvasPosition(255); got != 264 {
		t.Fatalf("decode canvas position 255 = %d, want 264", got)
	}
}
