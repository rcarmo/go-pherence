package main

import (
	"context"
	"encoding/json"
	"errors"
	"net/http/httptest"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model"
	"github.com/rcarmo/go-pherence/runtime/kv"
	"github.com/rcarmo/go-pherence/tensor"
)

func TestHandleChatCompletionsRejectsUnsupportedTemperature(t *testing.T) {
	s := &Server{}
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"hello"}],"temperature":0.7}`))
	rr := httptest.NewRecorder()
	s.handleChatCompletions(rr, req)
	if rr.Code != 400 {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), "unsupported sampling control: temperature") {
		t.Fatalf("unexpected error: %s", rr.Body.String())
	}
}

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

func TestHandleHealthReportsTurboQuantPolicyError(t *testing.T) {
	s := &Server{modelID: "qwen-reap", maxCtx: 16, cacheTypeK: "turbo9", cacheTypeV: "turbo2", kvResidual: 2, cpuModel: &model.LlamaModel{Config: model.LlamaConfig{NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HiddenSize: 4, HeadDim: 4, MaxSeqLen: 8}}, presets: map[string]ModelPreset{"qwen-reap": {ID: "qwen-reap"}}}
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
	if !ok || tq["error"] == nil {
		t.Fatalf("missing turboquant policy error: %+v", resp)
	}
	if tq["enabled"] != nil || tq["estimated_kv_bytes"] != nil || tq["simd_rotation"] != nil {
		t.Fatalf("malformed policy should not report normal TurboQuant details: %+v", tq)
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

func TestHandleChatCompletionsGemma4PreparedPromptAccounting(t *testing.T) {
	s := &Server{cpuModel: newGemma4ZeroLayerServerModel(), tok: newGemma4ZeroLayerServerTokenizer(), modelID: "gemma4-test", presets: map[string]ModelPreset{}, kvResidual: -1}
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"x"}],"max_tokens":4}`))
	rr := httptest.NewRecorder()
	s.handleChatCompletions(rr, req)
	if rr.Code != 200 {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	var resp ChatCompletionResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Model != "gemma4-test" || resp.Choices[0].Message.Content != "A" || resp.Choices[0].FinishReason != "stop" {
		t.Fatalf("unexpected completion response: %+v", resp)
	}
	if resp.Usage.PromptTokens != 2 || resp.Usage.CompletionTokens != 1 || resp.Usage.TotalTokens != 3 {
		t.Fatalf("unexpected usage accounting: %+v", resp.Usage)
	}
}

func TestGenerateGemma4CPUInvokesEmitImmediately(t *testing.T) {
	s := &Server{}
	rt := serverRuntime{cpuModel: newGemma4ZeroLayerServerModel(), tok: newGemma4ZeroLayerServerTokenizer(), modelID: "gemma4-test"}
	var emitted []string
	res, err := s.generate(context.Background(), rt, []int{0}, 4, func(token int, text string) bool {
		emitted = append(emitted, text)
		return true
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(emitted, []string{"A"}) {
		t.Fatalf("emitted=%v want [A]", emitted)
	}
	if res.PromptTokens != 2 || res.CompletionTokens != 1 || res.Text != "A" || res.FinishReason != "stop" {
		t.Fatalf("unexpected generation result: %+v", res)
	}
}

func TestGenerateGemma4CPUChecksContextBetweenDecodeSteps(t *testing.T) {
	fake := &fakeInferenceSession{
		prefill: model.PrefillResult{ConsumedTokens: 9, Position: 9, ReadyToDecode: true},
		steps:   []model.DecodeResult{{Token: 1, Position: 10, Generated: 1}, {Token: 1, Position: 11, Generated: 2}},
	}
	s := &Server{newGemmaSession: func(m *model.LlamaModel, opts model.SessionOptions) (model.InferenceSession, error) {
		if opts.Backend != model.InferenceBackendSIMD {
			t.Fatalf("backend=%s want simd", opts.Backend)
		}
		if opts.MaxTokens != 4 {
			t.Fatalf("max_tokens=%d want 4", opts.MaxTokens)
		}
		if !containsInt(opts.StopTokenIDs, 0) {
			t.Fatalf("stop tokens=%v want EOS-like token 0", opts.StopTokenIDs)
		}
		return fake, nil
	}}
	rt := serverRuntime{cpuModel: &model.LlamaModel{Config: model.LlamaConfig{ModelType: "gemma4_text"}}, tok: &tokenizer.Tokenizer{InvVocab: map[int]string{1: "A"}}, modelID: "gemma4-test"}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	emits := 0
	res, err := s.generate(ctx, rt, []int{7, 8}, 4, func(token int, text string) bool {
		emits++
		cancel()
		return true
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err=%v want context canceled", err)
	}
	if !reflect.DeepEqual(fake.prefillTokens, []int{7, 8}) {
		t.Fatalf("prefill tokens=%v want [7 8]", fake.prefillTokens)
	}
	if fake.decodeCalls != 1 {
		t.Fatalf("decode calls=%d want 1", fake.decodeCalls)
	}
	if !fake.closed {
		t.Fatal("session was not closed")
	}
	if emits != 1 || res.PromptTokens != 9 || res.CompletionTokens != 1 || res.Text != "A" {
		t.Fatalf("unexpected cancellation result emits=%d result=%+v", emits, res)
	}
}

type fakeInferenceSession struct {
	prefill       model.PrefillResult
	steps         []model.DecodeResult
	prefillTokens []int
	decodeCalls   int
	closed        bool
}

func (f *fakeInferenceSession) Backend() model.InferenceBackend { return model.InferenceBackendSIMD }

func (f *fakeInferenceSession) PrefillChunk(tokens []int) (model.PrefillResult, error) {
	f.prefillTokens = append([]int(nil), tokens...)
	return f.prefill, nil
}

func (f *fakeInferenceSession) DecodeStep() (model.DecodeResult, error) {
	if f.decodeCalls >= len(f.steps) {
		return model.DecodeResult{Finished: true}, nil
	}
	step := f.steps[f.decodeCalls]
	f.decodeCalls++
	return step, nil
}

func (f *fakeInferenceSession) Checkpoint() (model.SessionCheckpoint, error) { return nil, nil }

func (f *fakeInferenceSession) Restore(model.SessionCheckpoint) error { return nil }

func (f *fakeInferenceSession) Finished() (bool, model.FinishReason) {
	return false, model.FinishReasonNone
}

func (f *fakeInferenceSession) Close() error {
	f.closed = true
	return nil
}

func newGemma4ZeroLayerServerModel() *model.LlamaModel {
	return &model.LlamaModel{
		Config: model.LlamaConfig{ModelType: "gemma4_text", BOSTokenID: 2, VocabSize: 3, HiddenSize: 2, NumLayers: 0, NumHeads: 1, NumKVHeads: 1, HeadDim: 2, RMSNormEps: 0},
		EmbedTokens: tensor.FromFloat32([]float32{
			1, 0,
			0, 1,
			1, 1,
		}, []int{3, 2}),
		Norm: tensor.Ones([]int{2}),
		LMHead: tensor.FromFloat32([]float32{
			0, 1,
			1, 1,
			1, 0,
		}, []int{3, 2}),
	}
}

func newGemma4ZeroLayerServerTokenizer() *tokenizer.Tokenizer {
	return &tokenizer.Tokenizer{Vocab: map[string]int{"x": 0, "A": 1, "<bos>": 2}, InvVocab: map[int]string{0: "x", 1: "A", 2: "<bos>"}}
}

func containsInt(xs []int, want int) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
