package diffusiongemma

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

type fullCanvasTopLogitFixture struct {
	Reference    string                `json:"reference"`
	PromptIDs    []int                 `json:"prompt_ids"`
	CanvasToken  int                   `json:"canvas_token"`
	CanvasLength int                   `json:"canvas_length"`
	Row          int                   `json:"row"`
	TopLogits    []fullCanvasLogitRank `json:"top_logits"`
}

type fullCanvasLogitRank struct {
	ID int     `json:"id"`
	V  float32 `json:"v"`
}

func loadFullCanvasTopLogitFixture(t *testing.T) fullCanvasTopLogitFixture {
	t.Helper()
	path := filepath.Join("testdata", "gguf_prompt105_canvas236743_fullcanvas_toplogits.json")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var f fullCanvasTopLogitFixture
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	if f.Reference == "" || len(f.PromptIDs) == 0 || f.CanvasLength != 256 || len(f.TopLogits) == 0 {
		t.Fatalf("bad full-canvas fixture: %+v", f)
	}
	return f
}

func TestLlamaCppFullCanvasTopLogitFixture(t *testing.T) {
	f := loadFullCanvasTopLogitFixture(t)
	if f.PromptIDs[0] != 105 || f.CanvasToken != 236743 || f.Row != 0 {
		t.Fatalf("unexpected full-canvas fixture identity: %+v", f)
	}
	want := fullCanvasLogitRank{ID: 236743, V: 28.092376708984375}
	got := f.TopLogits[0]
	if got.ID != want.ID || math.Abs(float64(got.V-want.V)) > 1e-6 {
		t.Fatalf("fixture top=%+v want %+v", got, want)
	}
}

func TestLocalGGUFFullCanvasTopTokenMatchesLlamaCppFixture(t *testing.T) {
	if os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_FULL_CANVAS_GOLDEN") != "1" {
		t.Skip("set GO_PHERENCE_DIFFUSIONGEMMA_FULL_CANVAS_GOLDEN=1 for slow local full-canvas GGUF comparison")
	}
	ggufPath := filepath.Join("..", "..", "..", "llama.cpp", "models", "diffusiongemma-gguf", "diffusiongemma-26B-A4B-it-Q4_K_M.gguf")
	if _, err := os.Stat(ggufPath); err != nil {
		t.Skip("local DiffusionGemma GGUF Q4_K_M reference not present")
	}
	f := loadFullCanvasTopLogitFixture(t)
	den := openLocalGGUFTinyGoldenDenoiser(t)
	canvas := make([]int, f.CanvasLength)
	for i := range canvas {
		canvas[i] = f.CanvasToken
	}
	out, err := den.Denoise(ForwardInput{PromptIDs: f.PromptIDs, Canvas: canvas, Step: 1, SCTempInv: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Logits) <= f.Row {
		t.Fatalf("logits rows=%d want row %d", len(out.Logits), f.Row)
	}
	gotTop := topFiniteLogits(out.Logits[f.Row], 5)
	wantTop := f.TopLogits[0]
	if gotTop[0].id != wantTop.ID {
		t.Fatalf("full-canvas top token=%d want %d all=%v llama=%v", gotTop[0].id, wantTop.ID, gotTop, f.TopLogits)
	}
	// Keep the current full-canvas numerical gap visible but bounded until the
	// remaining all-layer logit parity work closes it. This prevents regressing
	// the actual llama.cpp graph top token while avoiding the invalid one-row gate.
	if diff := math.Abs(float64(gotTop[0].v - wantTop.V)); diff > 0.12 {
		t.Fatalf("full-canvas top logit diff=%g got=%g llama=%g all=%v", diff, gotTop[0].v, wantTop.V, gotTop)
	}
}
