package diffusiongemma

import (
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/rcarmo/go-pherence/loader/gguf"
)

type logitRank struct {
	id int
	v  float32
}

func topFiniteLogits(row []float32, n int) []logitRank {
	out := make([]logitRank, 0, len(row))
	for i, v := range row {
		if !math.IsInf(float64(v), -1) && !math.IsNaN(float64(v)) {
			out = append(out, logitRank{id: i, v: v})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].v > out[j].v })
	if len(out) > n {
		out = out[:n]
	}
	return out
}

func TestTopFiniteLogitsFiltersAndSorts(t *testing.T) {
	got := topFiniteLogits([]float32{1, float32(math.Inf(-1)), 3, float32(math.NaN()), 2}, 2)
	want := []logitRank{{id: 2, v: 3}, {id: 4, v: 2}}
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d got=%v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%+v want %+v all=%v", i, got[i], want[i], got)
		}
	}
}

func assertTopLogits(t *testing.T, row []float32, want []logitRank, context string) {
	t.Helper()
	got := topFiniteLogits(row, len(want))
	if len(got) != len(want) {
		t.Fatalf("%s top logits len=%d want %d", context, len(got), len(want))
	}
	for i := range want {
		if got[i].id != want[i].id || math.Abs(float64(got[i].v-want[i].v)) > 1e-3 {
			t.Fatalf("%s top[%d]=(%d,%g) want (%d,%g); all=%v", context, i, got[i].id, got[i].v, want[i].id, want[i].v, got)
		}
	}
}

func assertSelfConditioningLen(t *testing.T, got []float32, want int) {
	t.Helper()
	if len(got) != want {
		t.Fatalf("self-conditioning len=%d want %d", len(got), want)
	}
}

func assertGeneratedIDs(t *testing.T, got, want []int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("generated len=%d want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("generated[%d]=%d want %d all=%v", i, got[i], want[i], got)
		}
	}
}

func TestAssertCanvasStepsAcceptsExpectedMetadata(t *testing.T) {
	steps := []CanvasStep{{Step: 2, Accepted: 1, MeanEntropy: 2.25}, {Step: 1, Accepted: 1, MeanEntropy: 1.25}}
	assertCanvasSteps(t, steps, []float64{2.25, 1.25})
}

func assertCanvasSteps(t *testing.T, steps []CanvasStep, wantEntropy []float64) {
	t.Helper()
	if len(steps) != len(wantEntropy) {
		t.Fatalf("steps=%d want %d: %+v", len(steps), len(wantEntropy), steps)
	}
	for i := range wantEntropy {
		step := steps[i]
		if step.Step != len(wantEntropy)-i || step.Accepted != 1 || step.Stopped {
			t.Fatalf("bad step[%d] metadata: %+v", i, step)
		}
		if math.Abs(step.MeanEntropy-wantEntropy[i]) > 1e-6 {
			t.Fatalf("step[%d] entropy=%g want %g", i, step.MeanEntropy, wantEntropy[i])
		}
	}
}

func TestAssertCanvasStepDiagnosticsAcceptsExpectedMetadata(t *testing.T) {
	canvases := []CanvasResult{
		{Steps: []CanvasStep{{Step: 1, Accepted: 1, MeanEntropy: 1.25}}},
		{Steps: []CanvasStep{{Step: 2, Accepted: 1, MeanEntropy: 2.5}, {Step: 1, Accepted: 1, MeanEntropy: 1.5}}},
	}
	assertCanvasStepDiagnostics(t, canvases, [][]float64{{1.25}, {2.5, 1.5}})
}

func assertCanvasStepDiagnostics(t *testing.T, canvases []CanvasResult, wantEntropy [][]float64) {
	t.Helper()
	if len(canvases) != len(wantEntropy) {
		t.Fatalf("canvases=%d want %d: %+v", len(canvases), len(wantEntropy), canvases)
	}
	for c := range wantEntropy {
		if len(canvases[c].Steps) != len(wantEntropy[c]) {
			t.Fatalf("canvas=%d steps=%d want %d: %+v", c, len(canvases[c].Steps), len(wantEntropy[c]), canvases[c])
		}
		for s := range wantEntropy[c] {
			step := canvases[c].Steps[s]
			if step.Step != len(wantEntropy[c])-s || step.Accepted != 1 || step.Stopped {
				t.Fatalf("bad canvas=%d step=%d metadata: %+v", c, s, step)
			}
			if math.Abs(step.MeanEntropy-wantEntropy[c][s]) > 1e-6 {
				t.Fatalf("canvas=%d step=%d entropy=%g want %g", c, s, step.MeanEntropy, wantEntropy[c][s])
			}
		}
	}
}

func openLocalGGUFTinyGoldenDenoiser(t *testing.T) *TextDenoiser {
	t.Helper()
	t.Skip("local GGUF golden tests are heavy CPU/SIMD reference fixtures; replace/augment with GPU prompt golden fixtures before enabling by default")
	ggufPath := filepath.Join("..", "..", "..", "llama.cpp", "models", "diffusiongemma-gguf", "diffusiongemma-26B-A4B-it-Q4_K_M.gguf")
	if _, err := os.Stat(ggufPath); err != nil {
		t.Skip("local DiffusionGemma GGUF Q4_K_M reference not present")
	}
	modelDir := filepath.Join("..", "..", "models", "diffusiongemma-26B-A4B-it-FP8")
	meta, err := LoadMetadata(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Open(ggufPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Close)
	weights, err := OpenTextWeightsFromGGUF(g, meta.Shape)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = weights.Close() })
	idx, err := BuildGGUFExpertIndex(g, meta.Shape.TextLayers, meta.Shape.NumExperts)
	if err != nil {
		t.Fatal(err)
	}
	den, err := NewTextDenoiserWithDispatcher(meta.Shape, weights, CPUDispatcher{GGUFExpertIndex: idx, FinalLogitSoftcapping: float32(meta.Config.TextConfig.FinalLogitSoftcapping), SkipEviction: true})
	if err != nil {
		t.Fatal(err)
	}
	return den
}

func TestLocalGGUFTinyForwardGoldenTopLogits(t *testing.T) {
	den := openLocalGGUFTinyGoldenDenoiser(t)
	out, err := den.Denoise(ForwardInput{PromptIDs: []int{105}, Canvas: []int{236743}, Step: 1, SCTempInv: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Logits) != 1 {
		t.Fatalf("logit rows=%d want 1", len(out.Logits))
	}
	assertTopLogits(t, out.Logits[0], []logitRank{
		{id: 236778, v: 20.713585},
		{id: 207, v: 20.196819},
		{id: 198, v: 19.681643},
		{id: 45518, v: 19.266382},
		{id: 236771, v: 19.006660},
	}, "one-token prompt")
}

func TestLocalGGUFTinyForwardMultiPromptGoldenTopLogits(t *testing.T) {
	den := openLocalGGUFTinyGoldenDenoiser(t)
	out, err := den.Denoise(ForwardInput{PromptIDs: []int{105, 107}, Canvas: []int{236743}, Step: 1, SCTempInv: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Logits) != 1 {
		t.Fatalf("logit rows=%d want 1", len(out.Logits))
	}
	assertTopLogits(t, out.Logits[0], []logitRank{
		{id: 236778, v: 22.443890},
		{id: 236770, v: 19.904573},
		{id: 236812, v: 19.614756},
		{id: 236771, v: 18.780748},
		{id: 236800, v: 18.739267},
	}, "multi-token prompt")
}

func TestLocalGGUFTinyForwardMultiPromptTwoCanvasGoldenTopLogits(t *testing.T) {
	den := openLocalGGUFTinyGoldenDenoiser(t)
	out, err := den.Denoise(ForwardInput{PromptIDs: []int{105, 107}, Canvas: []int{236743, 236744}, Step: 1, SCTempInv: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Logits) != 2 {
		t.Fatalf("logit rows=%d want 2", len(out.Logits))
	}
	wantRows := [][]logitRank{
		{
			{id: 236778, v: 22.451084},
			{id: 107, v: 22.304323},
			{id: 236770, v: 22.302599},
			{id: 236771, v: 21.603258},
			{id: 235, v: 17.858952},
		},
		{
			{id: 545, v: 26.048100},
			{id: 107, v: 24.764221},
			{id: 236743, v: 23.936533},
			{id: 236770, v: 23.909200},
			{id: 140, v: 23.907146},
		},
	}
	for row := range wantRows {
		assertTopLogits(t, out.Logits[row], wantRows[row], "multi-token prompt canvas row "+itoa(row))
	}
}

func TestLocalGGUFTinyGenerateCanvasMultiPromptGolden(t *testing.T) {
	den := openLocalGGUFTinyGoldenDenoiser(t)
	cfg := DefaultDenoisingConfig()
	cfg.MaxDenoisingSteps = 1
	res, err := GenerateCanvas(den, []int{105, 107}, cfg, 1, den.Shape.VocabSize, NewMT19937RNG(0))
	if err != nil {
		t.Fatal(err)
	}
	assertGeneratedIDs(t, res.Canvas, []int{77232})
	assertCanvasSteps(t, res.Steps, []float64{4.950567710225018})
}

func TestLocalGGUFTinyGenerateCanvasGolden(t *testing.T) {
	den := openLocalGGUFTinyGoldenDenoiser(t)
	cfg := DefaultDenoisingConfig()
	cfg.MaxDenoisingSteps = 1
	res, err := GenerateCanvas(den, []int{105}, cfg, 1, den.Shape.VocabSize, NewMT19937RNG(0))
	if err != nil {
		t.Fatal(err)
	}
	assertGeneratedIDs(t, res.Canvas, []int{199340})
	assertCanvasSteps(t, res.Steps, []float64{1.074905})
}

func TestLocalGGUFTinyGenerateCanvasMultiPromptSelfConditioningGolden(t *testing.T) {
	den := openLocalGGUFTinyGoldenDenoiser(t)
	cfg := DefaultDenoisingConfig()
	cfg.MaxDenoisingSteps = 2
	res, err := GenerateCanvas(den, []int{105, 107}, cfg, 2, den.Shape.VocabSize, NewMT19937RNG(0))
	if err != nil {
		t.Fatal(err)
	}
	assertGeneratedIDs(t, res.Canvas, []int{236789, 43567})
	assertCanvasSteps(t, res.Steps, []float64{2.2759929071969416, 1.27629939718456})
}

func TestLocalGGUFTinyGenerateCanvasSelfConditioningGolden(t *testing.T) {
	den := openLocalGGUFTinyGoldenDenoiser(t)
	cfg := DefaultDenoisingConfig()
	cfg.MaxDenoisingSteps = 2
	res, err := GenerateCanvas(den, []int{105}, cfg, 2, den.Shape.VocabSize, NewMT19937RNG(0))
	if err != nil {
		t.Fatal(err)
	}
	assertGeneratedIDs(t, res.Canvas, []int{236761, 236761})
	assertCanvasSteps(t, res.Steps, []float64{4.961812248958, 2.735337912732})
}

func TestLocalGGUFTinyGenerateTokenIDsMultiPromptFullReencodeGolden(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_DISABLE_INCREMENTAL_KV", "1")
	den := openLocalGGUFTinyGoldenDenoiser(t)
	eng, err := NewEngine(&Model{Shape: den.Shape, Denoising: DefaultDenoisingConfig()}, den)
	if err != nil {
		t.Fatal(err)
	}
	cfg := DefaultDenoisingConfig()
	cfg.MaxDenoisingSteps = 1
	res, err := eng.GenerateTokenIDs([]int{105, 107}, InferenceOptions{MaxNewTokens: 2, CanvasLength: 1, Seed: 0, Denoising: &cfg})
	if err != nil {
		t.Fatal(err)
	}
	assertGeneratedIDs(t, res.Generated, []int{77232, 236793})
	assertCanvasStepDiagnostics(t, res.Canvases, [][]float64{{4.950567710225018}, {4.950112969072517}})
}

func TestLocalGGUFTinyGenerateTokenIDsMultiPromptIncrementalGolden(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_REQUIRE_INCREMENTAL_KV", "1")
	den := openLocalGGUFTinyGoldenDenoiser(t)
	eng, err := NewEngine(&Model{Shape: den.Shape, Denoising: DefaultDenoisingConfig()}, den)
	if err != nil {
		t.Fatal(err)
	}
	cfg := DefaultDenoisingConfig()
	cfg.MaxDenoisingSteps = 1
	res, err := eng.GenerateTokenIDs([]int{105, 107}, InferenceOptions{MaxNewTokens: 2, CanvasLength: 1, Seed: 0, Denoising: &cfg})
	if err != nil {
		t.Fatal(err)
	}
	assertGeneratedIDs(t, res.Generated, []int{77232, 236793})
	assertCanvasStepDiagnostics(t, res.Canvases, [][]float64{{4.950567710225018}, {4.950112969072517}})
}

func TestLocalGGUFTinyGenerateTokenIDsIncrementalGolden(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_REQUIRE_INCREMENTAL_KV", "1")
	den := openLocalGGUFTinyGoldenDenoiser(t)
	eng, err := NewEngine(&Model{Shape: den.Shape, Denoising: DefaultDenoisingConfig()}, den)
	if err != nil {
		t.Fatal(err)
	}
	cfg := DefaultDenoisingConfig()
	cfg.MaxDenoisingSteps = 1
	res, err := eng.GenerateTokenIDs([]int{105}, InferenceOptions{MaxNewTokens: 2, CanvasLength: 1, Seed: 0, Denoising: &cfg})
	if err != nil {
		t.Fatal(err)
	}
	assertGeneratedIDs(t, res.Generated, []int{199340, 125184})
	assertCanvasStepDiagnostics(t, res.Canvases, [][]float64{{1.0749052617652735}, {5.5066835829321725}})
}

func TestLocalGGUFTinyGenerateTokenIDsMultiPromptSelfConditioningFullReencodeGolden(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_DISABLE_INCREMENTAL_KV", "1")
	den := openLocalGGUFTinyGoldenDenoiser(t)
	eng, err := NewEngine(&Model{Shape: den.Shape, Denoising: DefaultDenoisingConfig()}, den)
	if err != nil {
		t.Fatal(err)
	}
	cfg := DefaultDenoisingConfig()
	cfg.MaxDenoisingSteps = 2
	res, err := eng.GenerateTokenIDs([]int{105, 107}, InferenceOptions{MaxNewTokens: 2, CanvasLength: 1, Seed: 0, Denoising: &cfg})
	if err != nil {
		t.Fatal(err)
	}
	assertGeneratedIDs(t, res.Generated, []int{101, 1})
	assertCanvasStepDiagnostics(t, res.Canvases, [][]float64{
		{4.950567710225018, 1.1961042527966028},
		{2.1527621649538347, 1.4968717162508307},
	})
}

func TestLocalGGUFTinyGenerateTokenIDsMultiPromptWithSelfConditioningGolden(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_REQUIRE_INCREMENTAL_KV", "1")
	den := openLocalGGUFTinyGoldenDenoiser(t)
	eng, err := NewEngine(&Model{Shape: den.Shape, Denoising: DefaultDenoisingConfig()}, den)
	if err != nil {
		t.Fatal(err)
	}
	cfg := DefaultDenoisingConfig()
	cfg.MaxDenoisingSteps = 2
	res, err := eng.GenerateTokenIDs([]int{105, 107}, InferenceOptions{MaxNewTokens: 2, CanvasLength: 1, Seed: 0, Denoising: &cfg})
	if err != nil {
		t.Fatal(err)
	}
	assertGeneratedIDs(t, res.Generated, []int{101, 1})
	assertCanvasStepDiagnostics(t, res.Canvases, [][]float64{
		{4.950567710225018, 1.1961042527966028},
		{2.1527621649538347, 1.4968717162508307},
	})
}

func TestLocalGGUFTinyGenerateTokenIDsWithSelfConditioningGolden(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_REQUIRE_INCREMENTAL_KV", "1")
	den := openLocalGGUFTinyGoldenDenoiser(t)
	eng, err := NewEngine(&Model{Shape: den.Shape, Denoising: DefaultDenoisingConfig()}, den)
	if err != nil {
		t.Fatal(err)
	}
	cfg := DefaultDenoisingConfig()
	cfg.MaxDenoisingSteps = 2
	res, err := eng.GenerateTokenIDs([]int{105}, InferenceOptions{MaxNewTokens: 2, CanvasLength: 1, Seed: 0, Denoising: &cfg})
	if err != nil {
		t.Fatal(err)
	}
	assertGeneratedIDs(t, res.Generated, []int{236783, 682})
	assertCanvasStepDiagnostics(t, res.Canvases, [][]float64{
		{1.0749052617652735, 1.8569853137856418},
		{4.704220912177734, 2.924562126318235},
	})
}

func TestLocalGGUFMultiPromptTwoCanvasSelfConditioningGoldenTopLogits(t *testing.T) {
	den := openLocalGGUFTinyGoldenDenoiser(t)
	prompt := []int{105, 107}
	canvas := []int{236743, 236744}
	first, err := den.Denoise(ForwardInput{PromptIDs: prompt, Canvas: canvas, Step: 2, SCTempInv: 1.25})
	if err != nil {
		t.Fatal(err)
	}
	assertSelfConditioningLen(t, first.SelfConditioning, den.Shape.TextHiddenSize*len(canvas))
	second, err := den.Denoise(ForwardInput{PromptIDs: prompt, Canvas: canvas, Step: 1, SCTempInv: 2.5, SelfConditioning: first.SelfConditioning})
	if err != nil {
		t.Fatal(err)
	}
	if len(second.Logits) != 2 {
		t.Fatalf("logit rows=%d want 2", len(second.Logits))
	}
	assertTopLogits(t, second.Logits[0], []logitRank{
		{id: 107, v: 26.895899},
		{id: 236778, v: 26.532902},
		{id: 101, v: 24.779585},
		{id: 1, v: 24.573310},
		{id: 100, v: 24.572369},
	}, "multi-token two-canvas self-conditioning row 0")
	assertTopLogits(t, second.Logits[1], []logitRank{
		{id: 545, v: 25.287770},
		{id: 236778, v: 25.129667},
		{id: 107, v: 23.926970},
		{id: 578, v: 23.687147},
		{id: 238204, v: 23.678928},
	}, "multi-token two-canvas self-conditioning row 1")
}

func TestLocalGGUFMultiPromptSelfConditioningGoldenTopLogits(t *testing.T) {
	den := openLocalGGUFTinyGoldenDenoiser(t)
	prompt := []int{105, 107}
	canvas := []int{236743}
	first, err := den.Denoise(ForwardInput{PromptIDs: prompt, Canvas: canvas, Step: 2, SCTempInv: 1.25})
	if err != nil {
		t.Fatal(err)
	}
	assertSelfConditioningLen(t, first.SelfConditioning, den.Shape.TextHiddenSize)
	second, err := den.Denoise(ForwardInput{PromptIDs: prompt, Canvas: canvas, Step: 1, SCTempInv: 2.5, SelfConditioning: first.SelfConditioning})
	if err != nil {
		t.Fatal(err)
	}
	assertTopLogits(t, second.Logits[0], []logitRank{
		{id: 236778, v: 24.006075},
		{id: 1, v: 22.305845},
		{id: 236800, v: 21.557951},
		{id: 107, v: 21.380968},
		{id: 101, v: 21.325540},
	}, "multi-token self-conditioning")
}

func TestLocalGGUFSelfConditioningGoldenTopLogits(t *testing.T) {
	den := openLocalGGUFTinyGoldenDenoiser(t)
	prompt := []int{105}
	canvas := []int{236743}
	first, err := den.Denoise(ForwardInput{PromptIDs: prompt, Canvas: canvas, Step: 2, SCTempInv: 1.25})
	if err != nil {
		t.Fatal(err)
	}
	assertSelfConditioningLen(t, first.SelfConditioning, den.Shape.TextHiddenSize)
	second, err := den.Denoise(ForwardInput{PromptIDs: prompt, Canvas: canvas, Step: 1, SCTempInv: 2.5, SelfConditioning: first.SelfConditioning})
	if err != nil {
		t.Fatal(err)
	}
	assertTopLogits(t, second.Logits[0], []logitRank{
		{id: 1018, v: 26.607105},
		{id: 1, v: 25.673777},
		{id: 236779, v: 25.312618},
		{id: 236761, v: 25.262501},
		{id: 236778, v: 24.773499},
	}, "one-token self-conditioning")
}
