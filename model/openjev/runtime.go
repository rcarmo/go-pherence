// Package openjev implements the native text NLI contract of AlexWortega/openjev.
package openjev

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model/qwen"
)

const (
	ModelPin = "4395b29714015162db6112de91c35688e6e42717"
	ModelID  = "AlexWortega/openjev/qwen3.5-0.8b-nli-v2s-long"
	Template = "Premise: {premise}\nHypothesis: {hypothesis}"
)

var Labels = [...]string{"contradiction", "entailment", "neutral"}

type rawTensor struct {
	raw   []byte
	dtype string
	shape []int
}

type Runtime struct {
	Bundle          *qwen.Qwen35NativeMTPBundle
	Tokenizer       *tokenizer.Tokenizer
	embedding, head rawTensor
	norm, rope      []float32
	maxInput        int
}

type config struct {
	Architectures []string                                                               `json:"architectures"`
	ModelType     string                                                                 `json:"model_type"`
	NLITemplate   string                                                                 `json:"nli_template"`
	IDToLabel     map[string]string                                                      `json:"id2label"`
	ProblemType   string                                                                 `json:"problem_type"`
	Text          struct{ HiddenSize, VocabSize, NumHiddenLayers, IntermediateSize int } `json:"text_config"`
}

func Load(dir string, maxInput int) (*Runtime, error)        { return load(dir, maxInput, true) }
func LoadFixture(dir string, maxInput int) (*Runtime, error) { return load(dir, maxInput, false) }
func load(dir string, maxInput int, released bool) (*Runtime, error) {
	if maxInput < 1 || maxInput > 4096 {
		return nil, fmt.Errorf("openjev: max input tokens must be 1..4096")
	}
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("openjev: read config: %w", err)
	}
	var cfg config
	if err = json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("openjev: parse config: %w", err)
	}
	if len(cfg.Architectures) != 1 || cfg.Architectures[0] != "Qwen3_5ForSequenceClassification" || cfg.ModelType != "qwen3_5" || cfg.NLITemplate != Template || cfg.ProblemType != "single_label_classification" {
		return nil, fmt.Errorf("openjev: unsupported checkpoint contract")
	}
	for i, label := range Labels {
		if cfg.IDToLabel[fmt.Sprint(i)] != label {
			return nil, fmt.Errorf("openjev: invalid label order")
		}
	}
	bundle, err := qwen.LoadQwen35NativeMTPBundleFromDir(dir)
	if err != nil {
		return nil, err
	}
	fail := func(e error) (*Runtime, error) { _ = bundle.Close(); return nil, e }
	m := bundle.Meta
	if m.ModelType != "qwen3_5_text" || m.HiddenSize < 1 || m.NumHiddenLayers < 1 || m.VocabSize < 1 || len(m.LayerTypes) != m.NumHiddenLayers {
		return fail(fmt.Errorf("openjev: unsupported Qwen3.5 text architecture"))
	}
	if released && (m.HiddenSize != 1024 || m.NumHiddenLayers != 24 || m.VocabSize != 248320 || m.IntermediateSize != 3584 || m.NumAttentionHeads != 8 || m.NumKeyValueHeads != 2 || m.HeadDim != 256) {
		return fail(fmt.Errorf("openjev: checkpoint is not the pinned 0.8B architecture"))
	}
	src := bundle.TensorSource()
	if src == nil {
		return fail(fmt.Errorf("openjev: missing tensor source"))
	}
	get := func(names ...string) (rawTensor, error) {
		var last error
		for _, name := range names {
			raw, dtype, shape, e := src.GetRaw(name)
			if e == nil {
				return rawTensor{raw, dtype, shape}, nil
			}
			last = e
		}
		return rawTensor{}, last
	}
	emb, err := get("model.language_model.embed_tokens.weight", "model.embed_tokens.weight")
	if err != nil {
		return fail(err)
	}
	head, err := get("score.weight")
	if err != nil {
		return fail(err)
	}
	if len(emb.shape) != 2 || emb.shape[0] != m.VocabSize || emb.shape[1] != m.HiddenSize || len(head.shape) != 2 || head.shape[0] != len(Labels) || head.shape[1] != m.HiddenSize {
		return fail(fmt.Errorf("openjev: embedding/head shape mismatch"))
	}
	norm, err := src.Get("model.language_model.norm.weight", []int{m.HiddenSize})
	if err != nil {
		norm, err = src.Get("model.norm.weight", []int{m.HiddenSize})
	}
	if err != nil {
		return fail(err)
	}
	tok, err := tokenizer.LoadWithConfig(dir)
	if err != nil {
		return fail(fmt.Errorf("openjev: load tokenizer: %w", err))
	}
	ropeMax := m.MaxPositionEmbeddings
	if ropeMax <= 0 || ropeMax > maxInput {
		ropeMax = maxInput
	}
	return &Runtime{Bundle: bundle, Tokenizer: tok, embedding: emb, head: head, norm: norm.Data(), rope: qwen.NewQwen35RoPEFreqs(m, ropeMax), maxInput: maxInput}, nil
}
func (r *Runtime) Close() error {
	if r == nil || r.Bundle == nil {
		return nil
	}
	return r.Bundle.Close()
}
func Render(premise, hypothesis string) (string, error) {
	premise = strings.TrimSpace(premise)
	hypothesis = strings.TrimSpace(hypothesis)
	if premise == "" || hypothesis == "" {
		return "", fmt.Errorf("openjev: premise and hypothesis must be nonempty")
	}
	return "Premise: " + premise + "\nHypothesis: " + hypothesis, nil
}
func (r *Runtime) Tokenize(premise, hypothesis string) ([]int, error) {
	if r == nil || r.Tokenizer == nil {
		return nil, fmt.Errorf("openjev: nil runtime")
	}
	text, err := Render(premise, hypothesis)
	if err != nil {
		return nil, err
	}
	ids := r.Tokenizer.Encode(text)
	if len(ids) == 0 {
		return nil, fmt.Errorf("openjev: empty token sequence")
	}
	if len(ids) > r.maxInput {
		ids = ids[:r.maxInput]
	}
	return ids, nil
}
func (r *Runtime) ScoreTokenIDs(ids []int) ([]float32, error) {
	if r == nil || r.Bundle == nil || len(ids) == 0 || len(ids) > r.maxInput {
		return nil, fmt.Errorf("openjev: invalid token sequence")
	}
	inputs := make([][]float32, len(ids))
	for i, id := range ids {
		row, e := r.embeddingRow(id)
		if e != nil {
			return nil, e
		}
		inputs[i] = row
	}
	state, e := r.Bundle.NewForwardState()
	if e != nil {
		return nil, e
	}
	outs, _, e := r.Bundle.Base.ForwardChunkLayerStreamed(inputs, state, r.rope, 1e-6, r.Bundle.Meta)
	if e != nil {
		return nil, e
	}
	h := outs[len(outs)-1]
	rmsNormZeroCentered(h, r.norm, 1e-6)
	roundBF16(h)
	out := make([]float32, len(Labels))
	for i := range out {
		out[i], e = r.headRowDot(i, h)
		if e != nil {
			return nil, e
		}
	}
	return out, nil
}
func (r *Runtime) Score(premise, hypothesis string) (Prediction, error) {
	ids, e := r.Tokenize(premise, hypothesis)
	if e != nil {
		return Prediction{}, e
	}
	logits, e := r.ScoreTokenIDs(ids)
	if e != nil {
		return Prediction{}, e
	}
	return Summarize(premise, hypothesis, ids, logits)
}
func (r *Runtime) embeddingRow(id int) ([]float32, error) {
	if id < 0 || id >= r.embedding.shape[0] {
		return nil, fmt.Errorf("openjev: token %d out of range", id)
	}
	cols := r.embedding.shape[1]
	out := make([]float32, cols)
	switch r.embedding.dtype {
	case "BF16":
		for i := range out {
			out[i] = half.BF16ToF32(binary.LittleEndian.Uint16(r.embedding.raw[(id*cols+i)*2:]))
		}
	case "F32":
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(r.embedding.raw[(id*cols+i)*4:]))
		}
	default:
		return nil, fmt.Errorf("openjev: unsupported embedding dtype %s", r.embedding.dtype)
	}
	return out, nil
}
func (r *Runtime) headRowDot(row int, x []float32) (float32, error) {
	if row < 0 || row >= r.head.shape[0] || len(x) != r.head.shape[1] {
		return 0, fmt.Errorf("openjev: invalid score-head row")
	}
	var sum float32
	switch r.head.dtype {
	case "BF16":
		off := row * len(x) * 2
		for i, v := range x {
			sum += v * half.BF16ToF32(binary.LittleEndian.Uint16(r.head.raw[off+i*2:]))
		}
	case "F32":
		off := row * len(x) * 4
		for i, v := range x {
			sum += v * math.Float32frombits(binary.LittleEndian.Uint32(r.head.raw[off+i*4:]))
		}
	default:
		return 0, fmt.Errorf("openjev: unsupported score-head dtype %s", r.head.dtype)
	}
	return sum, nil
}
func rmsNormZeroCentered(x, w []float32, eps float32) {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	scale := float32(1 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	for i := range x {
		x[i] *= scale * (1 + w[i])
	}
}
func roundBF16(x []float32) {
	for i, v := range x {
		bits := math.Float32bits(v)
		bias := uint32(0x7fff) + ((bits >> 16) & 1)
		x[i] = math.Float32frombits((bits + bias) & 0xffff0000)
	}
}
