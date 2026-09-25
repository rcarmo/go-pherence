package mojev

import (
	"bytes"
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
	return tensor.FromFloat32(data, shape), nil
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
	if s == nil || s.model == nil {
		return nil, fmt.Errorf("mojev: uninitialised text scorer")
	}
	if err := ValidateTextControls(req, tok); err != nil {
		return nil, err
	}
	encode := func(text string) ([]int, error) { return tok.Encode(text), nil }
	// Validate the entire public request before any model work. The model-free
	// assembler already owns the canonical order and per-kind validation rules.
	zeros := make([][]float64, len(req.Fields))
	fields := make([]TextField, len(req.Fields))
	menus := make([][]string, len(req.Fields))
	for f, field := range req.Fields {
		zeros[f] = make([]float64, len(field.Options))
		fields[f] = TextField{Name: field.ID, Description: field.Instructions, Options: field.SortedOptions}
		menus[f] = field.SortedOptions
	}
	if _, err := AssembleTextDecision(req, zeros, encode, 0, stateLimit, questionLimit); err != nil {
		return nil, err
	}
	packed, err := PackTextRows([]TextRow{{State: req.State, Menus: menus}}, fields, stateLimit, questionLimit, 0, encode)
	if err != nil {
		return nil, err
	}
	selectIDs := func(span []bool) []int {
		var ids []int
		for i, on := range span {
			if on {
				ids = append(ids, packed.IDs[0][i])
			}
		}
		return ids
	}
	row := EncodedRow{State: selectIDs(packed.State[0]), Questions: make([][]int, len(fields)), Candidates: make([][][]int, len(fields))}
	for f := range fields {
		row.Questions[f] = selectIDs(packed.Questions[0][f])
		for n, on := range packed.OptionMask[0][f] {
			if on {
				row.Candidates[f] = append(row.Candidates[f], selectIDs(packed.Candidates[0][f][n]))
			}
		}
	}
	logits, err := s.ScoreEncoded(row)
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
	return AssembleTextDecision(req, rows, encode, 0, stateLimit, questionLimit)
}
