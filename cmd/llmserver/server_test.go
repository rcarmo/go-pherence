package main

import (
	"encoding/json"
	"net/http/httptest"
	"runtime"
	"testing"

	"github.com/rcarmo/go-pherence/model"
	"github.com/rcarmo/go-pherence/runtime/kv"
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
	s := &Server{modelID: "qwen-reap", maxCtx: 16, cacheTypeK: "turbo4", cacheTypeV: "turbo2", kvResidual: 2, cpuModel: &model.LlamaModel{Config: model.LlamaConfig{NumLayers: 8, NumHeads: 2, NumKVHeads: 1, HiddenSize: 8, HeadDim: 4, MaxSeqLen: 32}}, presets: map[string]ModelPreset{"qwen-reap": {ID: "qwen-reap"}}}
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
	if !ok || tq["enabled"] != true || int(tq["kv_dim"].(float64)) != 4 || int(tq["full_kv_bytes"].(float64)) != 4096 {
		t.Fatalf("unexpected turboquant health: %+v", resp)
	}
	if est := int(tq["estimated_kv_bytes"].(float64)); est <= 0 || est > 4096 {
		t.Fatalf("unexpected turboquant estimate: %+v", tq)
	}
	if saved := int(tq["estimated_saved_kv_bytes"].(float64)); saved <= 0 || saved >= 4096 {
		t.Fatalf("unexpected turboquant saved bytes: %+v", tq)
	}
	if ratio := tq["estimated_kv_ratio"].(float64); ratio <= 0 || ratio >= 1 {
		t.Fatalf("unexpected turboquant ratio: %+v", tq)
	}
	cfg, enabled, err := kv.TurboQuantConfigFromCacheTypes("turbo4", "turbo2", 2)
	if err != nil || !enabled {
		t.Fatalf("test TurboQuant config: enabled=%v err=%v", enabled, err)
	}
	wantEstimate := kv.EstimateTurboQuantKV(8, 1, 4, 16, cfg, true, nil)
	estimated := int64(tq["estimated_kv_bytes"].(float64))
	scratch := int64(tq["estimated_scratch_bytes"].(float64))
	total := int64(tq["estimated_total_bytes"].(float64))
	if estimated != wantEstimate.EstimatedBytes || scratch != wantEstimate.EstimatedScratchBytes || total != wantEstimate.EstimatedTotalBytes {
		t.Fatalf("unexpected turboquant scratch/total bytes: %+v want=%+v", tq, wantEstimate)
	}
	if int(tq["kv_layers"].(float64)) != wantEstimate.KVLayers || int(tq["protected_layers"].(float64)) != wantEstimate.ProtectedLayers {
		t.Fatalf("unexpected turboquant layer accounting: %+v", tq)
	}
	caps := kv.RuntimeTurboQuantCapabilities()
	if tq["simd_arch"] != runtime.GOARCH || tq["simd_rotation"] != caps.Rotation || tq["simd_vec"] != caps.Vec {
		t.Fatalf("unexpected turboquant SIMD health: %+v caps=%+v", tq, caps)
	}
	if tq["simd_avx2"] != caps.AVX2 || tq["simd_neon"] != caps.NEON || tq["simd_rvv"] != caps.RVV {
		t.Fatalf("unexpected turboquant SIMD ISA health: %+v caps=%+v", tq, caps)
	}
}

func TestHandleHealthReportsREAPSummary(t *testing.T) {
	s := &Server{modelID: "qwen-reap", maxCtx: 16, kvResidual: -1, cpuModel: &model.LlamaModel{REAP: &model.REAPConfig{Enabled: true, PruneRatio: 0.2, Source: "filename_or_name", DefaultMask: map[int]bool{1: true}}}, presets: map[string]ModelPreset{"qwen-reap": {ID: "qwen-reap"}}}
	rr := httptest.NewRecorder()
	s.handleHealth(rr, httptest.NewRequest("GET", "/health", nil))
	if rr.Code != 200 {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	var resp map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	reap, ok := resp["reap"].(map[string]any)
	if !ok || reap["enabled"] != true || reap["source"] != "filename_or_name" || int(reap["default_experts"].(float64)) != 1 {
		t.Fatalf("unexpected REAP health: %+v", resp)
	}
}

func TestHandleHealthReportsREAPAndTurboQuantTogether(t *testing.T) {
	s := &Server{modelID: "qwen-reap", maxCtx: 16, cacheTypeK: "turbo4", cacheTypeV: "turbo2", kvResidual: 2, cpuModel: &model.LlamaModel{Config: model.LlamaConfig{NumLayers: 8, NumHeads: 2, NumKVHeads: 1, HiddenSize: 8, HeadDim: 4, MaxSeqLen: 32}, REAP: &model.REAPConfig{Enabled: true, PruneRatio: 0.2, Source: "filename_or_name"}}, presets: map[string]ModelPreset{"qwen-reap": {ID: "qwen-reap"}}}
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
	if !ok || tq["enabled"] != true || tq["simd_rotation"] != kv.RuntimeTurboQuantCapabilities().Rotation {
		t.Fatalf("missing turboquant health: %+v", resp)
	}
	if int64(tq["estimated_total_bytes"].(float64)) != int64(tq["estimated_kv_bytes"].(float64))+int64(tq["estimated_scratch_bytes"].(float64)) {
		t.Fatalf("bad combined TurboQuant total: %+v", tq)
	}
	if int(tq["kv_layers"].(float64)) != 8 || int(tq["protected_layers"].(float64)) != 4 {
		t.Fatalf("bad combined TurboQuant layer accounting: %+v", tq)
	}
	if reap, ok := resp["reap"].(map[string]any); !ok || reap["source"] != "filename_or_name" || reap["enabled"] != true {
		t.Fatalf("missing REAP health: %+v", resp)
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
