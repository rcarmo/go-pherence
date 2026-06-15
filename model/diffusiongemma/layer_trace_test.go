package diffusiongemma

import "testing"

func TestDiffusionGemmaLayerTraceRowEnv(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_LAYER_TRACE_ROW", "")
	if got := diffusionGemmaLayerTraceRow(); got != -1 {
		t.Fatalf("empty trace row=%d want -1", got)
	}
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_LAYER_TRACE_ROW", "28")
	if got := diffusionGemmaLayerTraceRow(); got != 28 {
		t.Fatalf("trace row=%d want 28", got)
	}
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_LAYER_TRACE_ROW", "bad")
	if got := diffusionGemmaLayerTraceRow(); got != -1 {
		t.Fatalf("bad trace row=%d want -1", got)
	}
}

func TestDiffusionGemmaLayerTraceOpsEnabled(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_LAYER_TRACE_OPS", "")
	if diffusionGemmaLayerTraceOpsEnabled() {
		t.Fatal("op trace should default off")
	}
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_LAYER_TRACE_OPS", "yes")
	if !diffusionGemmaLayerTraceOpsEnabled() {
		t.Fatal("op trace opt-in not honored")
	}
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_LAYER_TRACE_OPS", "false")
	if diffusionGemmaLayerTraceOpsEnabled() {
		t.Fatal("op trace false should disable")
	}
}

func TestTraceForwardRowBoundsAreSafe(t *testing.T) {
	traceForwardRow("test", 0, -1, ForwardScratch{}, 4)
	traceForwardRow("test", 0, 10, ForwardScratch{Hidden: make([]float32, 8)}, 4)
	traceForwardRow("test", 0, 1, ForwardScratch{Hidden: []float32{1, 2, 3, 4, 5, 6, 7, 8}}, 4)
}
