package mojev

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/loader/weights"
	"github.com/rcarmo/go-pherence/model/qwen"
	"github.com/rcarmo/go-pherence/tensor"
)

// TextScorer owns the released text weights decoded to F32. It runs candidates
// separately with fresh attention state. It does not implement image input or
// the released BF16 packed forward, and does not change RuntimeReady. RoPE
// positions restart at zero for each path, so sibling lengths/order cannot
// influence scores. This differs from the earlier absolute-position probe.
type TextScorer struct {
	model                 *qwen.Qwen35BaseModel
	head                  *HeadWeights
	embedding, norm, rope []float32
	meta                  loaderconfig.QwenNativeMTPMetadata
	eps                   float32
}

type mojevTextSource struct{ source weights.Source }

func (s mojevTextSource) Get(name string, shape []int) (*tensor.Tensor, error) {
	if !strings.HasPrefix(name, "model.") {
		return nil, fmt.Errorf("mojev: unsupported text tensor %s", name)
	}
	name = "encoder.language_model." + strings.TrimPrefix(name, "model.")
	_, dtype, actual, err := s.source.GetRaw(name)
	if err != nil {
		return nil, err
	}
	if dtype != "BF16" && dtype != "F32" {
		return nil, fmt.Errorf("mojev: unexpected dtype %s for %s", dtype, name)
	}
	if !slices.Equal(actual, shape) {
		return nil, fmt.Errorf("mojev: %s shape %v want %v", name, actual, shape)
	}
	data, actual, err := s.source.GetFloat32(name)
	if err != nil {
		return nil, err
	}
	if !slices.Equal(actual, shape) {
		return nil, fmt.Errorf("mojev: decoded shape mismatch")
	}
	size := 1
	for _, d := range shape {
		if d <= 0 || d > math.MaxInt/size {
			return nil, fmt.Errorf("mojev: invalid shape")
		}
		size *= d
	}
	if len(data) != size {
		return nil, fmt.Errorf("mojev: short tensor %s", name)
	}
	for _, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("mojev: nonfinite tensor %s", name)
		}
	}
	// Source.GetFloat32 transfers an owned conversion buffer. Move that buffer
	// into the tensor after validation rather than duplicating the full weight.
	// GetRaw remains borrowed and is used only for metadata checks above.
	return tensor.FromOwnedFloat32(data, shape), nil
}

// LoadTextScorer validates the pinned topology and loads owned F32 text/head
// weights. The source remains caller-owned and may be closed after success.
// Vision weights are neither needed nor loaded by this text-only path.
func LoadTextScorer(src weights.Source, configJSON []byte) (*TextScorer, error) {
	if src == nil {
		return nil, fmt.Errorf("mojev: nil weight source")
	}
	cfg, err := ReadConfig(bytes.NewReader(configJSON))
	if err != nil {
		return nil, err
	}
	var raw struct {
		Encoder json.RawMessage `json:"encoder_config"`
	}
	if err = json.Unmarshal(configJSON, &raw); err != nil {
		return nil, err
	}
	meta, err := loaderconfig.ParseQwenNativeMTPMetadata(raw.Encoder)
	if err != nil {
		return nil, err
	}
	var encoder struct {
		Text struct {
			Eps float32 `json:"rms_norm_eps"`
		} `json:"text_config"`
	}
	if err = json.Unmarshal(raw.Encoder, &encoder); err != nil {
		return nil, err
	}
	if encoder.Text.Eps <= 0 || math.IsNaN(float64(encoder.Text.Eps)) || math.IsInf(float64(encoder.Text.Eps), 0) {
		return nil, fmt.Errorf("mojev: invalid RMS epsilon")
	}
	if encoder.Text.Eps != 1e-6 || meta.RopeTheta != 10000000 || meta.PartialRotaryFactor != 0.25 || !meta.ZeroCenteredRMSNorm || meta.QuantBits != 0 {
		return nil, fmt.Errorf("mojev: unsupported text arithmetic or RoPE configuration")
	}
	meta.BF16Trajectory = false
	source := mojevTextSource{src}
	model, err := qwen.LoadQwen35BaseModelLayers(source, meta)
	if err != nil {
		return nil, err
	}
	emb, err := source.Get("model.embed_tokens.weight", []int{meta.VocabSize, meta.HiddenSize})
	if err != nil {
		return nil, err
	}
	norm, err := source.Get("model.norm.weight", []int{meta.HiddenSize})
	if err != nil {
		return nil, err
	}
	head, err := LoadHead(src, cfg.Hidden, cfg.Rank)
	if err != nil {
		return nil, err
	}
	return &TextScorer{model: model, head: head, embedding: emb.Data(), norm: norm.Data(), rope: qwen.NewQwen35RoPEFreqs(meta, 4096), meta: meta, eps: encoder.Text.Eps}, nil
}

func (s *TextScorer) encodeBranch(branch TextBranch) ([]float32, error) {
	return s.encodeBranchWith(branch, nil)
}

func (s *TextScorer) encodeBranchWith(branch TextBranch, fast *qwen.Qwen35SIMDBranch) ([]float32, error) {
	if s == nil || s.model == nil {
		return nil, fmt.Errorf("mojev: uninitialised text scorer")
	}
	h := s.meta.HiddenSize
	rows := make([][]float32, len(branch.IDs))
	for i, id := range branch.IDs {
		if id < 0 || id >= s.meta.VocabSize {
			return nil, fmt.Errorf("mojev: token outside vocabulary")
		}
		rows[i] = s.embedding[id*h : (id+1)*h]
	}
	if fast != nil {
		out := make([]float32, len(rows)*h)
		if err := fast.ForwardInto(out, rows, branch.StateLen, branch.QuestionLen, s.rope, s.eps); err != nil {
			return nil, err
		}
		s.normaliseFinal(out)
		return out, nil
	}
	// Local positions prevent sibling length or ordering from changing RoPE.
	positions := make([]int, len(rows))
	for i := range positions {
		positions[i] = i
	}
	hidden, err := s.model.ForwardTextBranch(rows, positions, branch.StateLen, branch.QuestionLen, s.rope, s.eps, s.meta)
	if err != nil {
		return nil, err
	}
	out := make([]float32, 0, len(rows)*h)
	for _, row := range hidden {
		var sum float32
		for _, v := range row {
			sum += v * v
		}
		inv := float32(1 / math.Sqrt(float64(sum/float32(h)+s.eps)))
		for i, v := range row {
			out = append(out, v*inv*(1+s.norm[i]))
		}
	}
	return out, nil
}

func (s *TextScorer) normaliseFinal(rows []float32) {
	h := s.meta.HiddenSize
	for t := 0; t < len(rows)/h; t++ {
		row := rows[t*h : (t+1)*h]
		var sum float32
		for _, v := range row {
			sum += v * v
		}
		inv := float32(1 / math.Sqrt(float64(sum/float32(h)+s.eps)))
		for i, v := range row {
			row[i] = v * inv * (1 + s.norm[i])
		}
	}
}

// ScoreEncoded computes real text-only encoder and head logits from one encoded
// row, with no shared request state and no supplied logits/hidden rows.
func (s *TextScorer) ScoreEncoded(row EncodedRow) ([][]float32, error) {
	if s == nil || s.model == nil {
		return nil, fmt.Errorf("mojev: uninitialised text scorer")
	}
	return ScoreBranchLocalText(row, s.head, s.encodeBranch)
}

// ScoreText validates reserved tokens and request ordering, runs the native
// text encoder, and returns owned public answers. Usage counts logical packed
// tokens, not the repeated ancestor work in per-candidate evaluation. Upstream
// prompts include small option menus: editing a menu therefore also edits that
// question's prompt. Cross-question isolation still holds; raw candidate-only
// isolation applies to ScoreEncoded with unchanged question tokens.
func (s *TextScorer) ScoreText(req TextRequest, tok *tokenizer.Tokenizer, stateLimit, questionLimit int) (*TextDecision, error) {
	return s.scoreText(req, tok, stateLimit, questionLimit, s.ScoreEncoded)
}

func (s *TextScorer) scoreText(req TextRequest, tok *tokenizer.Tokenizer, stateLimit, questionLimit int, score func(EncodedRow) ([][]float32, error)) (*TextDecision, error) {
	return s.scoreTextContext(context.Background(), req, tok, stateLimit, questionLimit, score)
}

func (s *TextScorer) scoreTextContext(ctx context.Context, req TextRequest, tok *tokenizer.Tokenizer, stateLimit, questionLimit int, score func(EncodedRow) ([][]float32, error)) (*TextDecision, error) {
	if ctx == nil {
		return nil, fmt.Errorf("mojev: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if s == nil || s.model == nil {
		return nil, fmt.Errorf("mojev: uninitialised text scorer")
	}
	return scoreTextContextWith(ctx, req, tok, stateLimit, questionLimit, score)
}

// Common tokenizer/answer orchestration has no dependency on CPU encoder
// weights. Accelerators supply a scorer callback and retain only host readout.
func scoreTextContextWith(ctx context.Context, req TextRequest, tok *tokenizer.Tokenizer, stateLimit, questionLimit int, score func(EncodedRow) ([][]float32, error)) (*TextDecision, error) {
	if ctx == nil {
		return nil, fmt.Errorf("mojev: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if score == nil {
		return nil, fmt.Errorf("mojev: nil scoring callback")
	}
	if err := ValidateTextControls(req, tok); err != nil {
		return nil, err
	}
	prepared, err := prepareTextDecision(req, nil)
	if err != nil {
		return nil, err
	}
	encode := func(text string) ([]int, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		ids := tok.Encode(text)
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return ids, nil
	}
	encoded, err := encodeTextRows([]TextRow{{State: prepared.req.State, Menus: prepared.menus}}, prepared.fields, stateLimit, questionLimit, encode)
	if err != nil {
		return nil, err
	}
	row := encoded[0]
	tokens := encodedRowTokenCount(row)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	logits, err := score(row)
	if err != nil {
		return nil, err
	}
	rows := make([][]float64, len(logits))
	for f, row := range logits {
		rows[f] = make([]float64, len(row))
		for n, v := range row {
			rows[f][n] = float64(v)
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	answer, err := prepared.assembleTokens(rows, tokens)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return answer, nil
}
