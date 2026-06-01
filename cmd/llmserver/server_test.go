package main

import (
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/rcarmo/go-pherence/model"
)

func TestHandleModelsListsPresets(t *testing.T) {
	s := &Server{
		modelID: "qwen-reap",
		created: 123,
		presets: map[string]ModelPreset{
			"qwen-reap": {ID: "qwen-reap", Path: "/models/qwen"},
			"glm-reap":  {ID: "glm-reap", Path: "/models/glm"},
		},
	}
	rr := httptest.NewRecorder()
	s.handleModels(rr, httptest.NewRequest("GET", "/v1/models", nil))
	if rr.Code != 200 {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	var resp ModelListResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Object != "list" || len(resp.Data) != 2 {
		t.Fatalf("unexpected models response: %+v", resp)
	}
	seen := map[string]bool{}
	for _, m := range resp.Data {
		seen[m.ID] = true
	}
	if !seen["qwen-reap"] || !seen["glm-reap"] {
		t.Fatalf("missing presets in response: %+v", resp.Data)
	}
}

func TestHandleHealthReportsTurboQuantPlan(t *testing.T) {
	s := &Server{modelID: "qwen-reap", maxCtx: 16, cacheTypeK: "turbo4", cacheTypeV: "turbo2", kvResidual: 2, cpuModel: &model.LlamaModel{Config: model.LlamaConfig{NumLayers: 4, NumHeads: 2, NumKVHeads: 1, HiddenSize: 8, HeadDim: 4, MaxSeqLen: 32}}, presets: map[string]ModelPreset{"qwen-reap": {ID: "qwen-reap"}}}
	rr := httptest.NewRecorder()
	s.handleHealth(rr, httptest.NewRequest("GET", "/health", nil))
	if rr.Code != 200 {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	var resp map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	tq, ok := resp["turboquant"].(map[string]any)
	if !ok || tq["enabled"] != true || int(tq["kv_dim"].(float64)) != 4 || int(tq["full_kv_bytes"].(float64)) != 2048 {
		t.Fatalf("unexpected turboquant health: %+v", resp)
	}
	if est := int(tq["estimated_kv_bytes"].(float64)); est <= 0 || est > 2048 {
		t.Fatalf("unexpected turboquant estimate: %+v", tq)
	}
}

func TestHandleHealthReportsRuntimeState(t *testing.T) {
	s := &Server{modelID: "qwen-reap", maxCtx: 32768, presets: map[string]ModelPreset{"qwen-reap": {ID: "qwen-reap"}}, kvResidual: -1}
	rr := httptest.NewRecorder()
	s.handleHealth(rr, httptest.NewRequest("GET", "/health", nil))
	if rr.Code != 200 {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	var resp map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp["status"] != "ok" || resp["model"] != "qwen-reap" || int(resp["ctx_size"].(float64)) != 32768 {
		t.Fatalf("unexpected health response: %+v", resp)
	}
}
