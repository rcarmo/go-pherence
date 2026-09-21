package decider

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

type rawTensor struct {
	raw   []byte
	dtype string
	shape []int
}

type Runtime struct {
	Bundle            *qwen.Qwen35NativeMTPBundle
	Tokenizer         *tokenizer.Tokenizer
	embedding         rawTensor
	norm, rope        []float32
	labels            []int
	maxInput          int
	Temperature       float32
	SchemaTemperature float32
	Version           string
	modelDir          string
}

type diskConfig struct {
	Temperature        float32 `json:"temperature"`
	SchemaTemperature  float32 `json:"temperature_schema_first"`
	Version            string  `json:"version"`
	Base               string  `json:"base"`
	MaxOptions         int     `json:"max_options"`
	MaxStateTokens     int     `json:"max_state_tokens"`
	SchemaFirst        bool    `json:"schema_first"`
	SchemaFirstTrained bool    `json:"schema_first_trained"`
	IsolatedLevels     bool    `json:"isolated_levels"`
}

func Load(dir string, maxInput int) (*Runtime, error) {
	return load(dir, maxInput, true)
}

// LoadFixture is for bounded synthetic tests and downstream architecture
// qualification. Production callers should use Load, which enforces the pinned
// 0.8B geometry in addition to the common tensor and sidecar checks.
func LoadFixture(dir string, maxInput int) (*Runtime, error) {
	return load(dir, maxInput, false)
}

func load(dir string, maxInput int, released bool) (*Runtime, error) {
	if maxInput < 1 || maxInput > 32768 {
		return nil, fmt.Errorf("decider: max input tokens must be 1..32768")
	}
	cfgRaw, err := os.ReadFile(filepath.Join(dir, "decider_config.json"))
	if err != nil {
		return nil, fmt.Errorf("decider: read config: %w", err)
	}
	var cfg diskConfig
	if err := json.Unmarshal(cfgRaw, &cfg); err != nil {
		return nil, fmt.Errorf("decider: parse config: %w", err)
	}
	if cfg.Temperature <= 0 || !isFinite(cfg.Temperature) || cfg.MaxOptions != MaxChoice || cfg.Base != "Qwen/Qwen3.5-0.8B-Base" || !cfg.IsolatedLevels {
		return nil, fmt.Errorf("decider: unsupported release contract")
	}
	bundle, err := qwen.LoadQwen35NativeMTPBundleFromDir(dir)
	if err != nil {
		return nil, err
	}
	fail := func(e error) (*Runtime, error) { _ = bundle.Close(); return nil, e }
	if bundle.Meta.ModelType != "qwen3_5_text" || bundle.Meta.HiddenSize < 1 || bundle.Meta.NumHiddenLayers < 1 || bundle.Meta.VocabSize < MaxChoice || len(bundle.Meta.LayerTypes) != bundle.Meta.NumHiddenLayers {
		return fail(fmt.Errorf("decider: unsupported Qwen3.5 architecture"))
	}
	if released && (bundle.Meta.HiddenSize != 1024 || bundle.Meta.NumHiddenLayers != 24 || bundle.Meta.VocabSize != 248320 || bundle.Meta.IntermediateSize != 3584 || bundle.Meta.NumAttentionHeads != 8 || bundle.Meta.NumKeyValueHeads != 2 || bundle.Meta.HeadDim != 256) {
		return fail(fmt.Errorf("decider: checkpoint is not the pinned 0.8B architecture"))
	}
	src := bundle.TensorSource()
	if src == nil {
		return fail(fmt.Errorf("decider: missing tensor source"))
	}
	var emb rawTensor
	for _, name := range []string{"model.language_model.embed_tokens.weight", "model.embed_tokens.weight"} {
		raw, dtype, shape, e := src.GetRaw(name)
		if e == nil {
			emb = rawTensor{raw: raw, dtype: dtype, shape: shape}
			break
		}
	}
	if len(emb.shape) != 2 || emb.shape[0] != bundle.Meta.VocabSize || emb.shape[1] != bundle.Meta.HiddenSize || (emb.dtype != "BF16" && emb.dtype != "F32") {
		return fail(fmt.Errorf("decider: invalid tied embedding/head tensor"))
	}
	norm, err := src.Get("model.language_model.norm.weight", []int{bundle.Meta.HiddenSize})
	if err != nil {
		norm, err = src.Get("model.norm.weight", []int{bundle.Meta.HiddenSize})
	}
	if err != nil {
		return fail(err)
	}
	tok, err := tokenizer.LoadWithConfig(dir)
	if err != nil {
		return fail(fmt.Errorf("decider: load tokenizer: %w", err))
	}
	labels, err := labelIDs(tok)
	if err != nil {
		return fail(err)
	}
	ropeMax := bundle.Meta.MaxPositionEmbeddings
	if ropeMax <= 0 || ropeMax > maxInput {
		ropeMax = maxInput
	}
	if cfg.SchemaTemperature <= 0 {
		cfg.SchemaTemperature = cfg.Temperature
	}
	return &Runtime{Bundle: bundle, Tokenizer: tok, embedding: emb, norm: norm.Data(), rope: qwen.NewQwen35RoPEFreqs(bundle.Meta, ropeMax), labels: labels, maxInput: maxInput, Temperature: cfg.Temperature, SchemaTemperature: cfg.SchemaTemperature, Version: cfg.Version, modelDir: dir}, nil
}

func labelIDs(tok *tokenizer.Tokenizer) ([]int, error) {
	if tok == nil {
		return nil, fmt.Errorf("decider: nil tokenizer")
	}
	labels := make([]int, 0, MaxChoice)
	for a := 'A'; a <= 'Z' && len(labels) < MaxChoice; a++ {
		name := string(a)
		ids := tok.Encode(name)
		if len(ids) == 1 {
			labels = append(labels, ids[0])
		}
	}
	for a := 'A'; a <= 'Z' && len(labels) < MaxChoice; a++ {
		for b := 'A'; b <= 'Z' && len(labels) < MaxChoice; b++ {
			ids := tok.Encode(string([]rune{a, b}))
			if len(ids) == 1 {
				labels = append(labels, ids[0])
			}
		}
	}
	if len(labels) != MaxChoice {
		return nil, fmt.Errorf("decider: tokenizer provides %d one-token labels, want %d", len(labels), MaxChoice)
	}
	seen := make(map[int]bool, len(labels))
	for _, id := range labels {
		if seen[id] {
			return nil, fmt.Errorf("decider: invalid label table")
		}
		seen[id] = true
	}
	return labels, nil
}

func (r *Runtime) Close() error {
	if r == nil || r.Bundle == nil {
		return nil
	}
	return r.Bundle.Close()
}

func (r *Runtime) LabelTokenIDs(n int) ([]int, error) {
	if r == nil || n < 1 || n > len(r.labels) {
		return nil, fmt.Errorf("decider: invalid label count")
	}
	return append([]int(nil), r.labels[:n]...), nil
}

func labelName(i int) string {
	if i < 26 {
		return string(rune('A' + i))
	}
	i -= 26
	return string([]rune{rune('A' + i/26), rune('A' + i%26)})
}

func BuildPrompt(state, question string, options []string) (string, error) {
	if strings.TrimSpace(state) == "" || strings.TrimSpace(question) == "" || len(options) < 2 || len(options) > MaxChoice {
		return "", fmt.Errorf("decider: invalid prompt inputs")
	}
	var b strings.Builder
	b.WriteString("Context:\n")
	b.WriteString(state)
	b.WriteString("\n\nQuestion: ")
	b.WriteString(question)
	b.WriteString("\nOptions:")
	for i, option := range options {
		if strings.TrimSpace(option) == "" {
			return "", fmt.Errorf("decider: empty option %d", i)
		}
		b.WriteString("\n(")
		b.WriteString(labelName(i))
		b.WriteString(") ")
		b.WriteString(option)
	}
	b.WriteString("\nAnswer: (")
	return b.String(), nil
}

func (r *Runtime) TokenizePrompt(text string) ([]int, error) {
	if r == nil || r.Tokenizer == nil {
		return nil, fmt.Errorf("decider: nil runtime")
	}
	ids := r.Tokenizer.Encode(text)
	return r.admitIDs(ids)
}

func (r *Runtime) admitIDs(ids []int) ([]int, error) {
	if len(ids) < 1 || len(ids) > r.maxInput {
		return nil, fmt.Errorf("decider: prompt has %d tokens; limit %d", len(ids), r.maxInput)
	}
	return ids, nil
}

// BuildPromptIDs preserves the released wide-label contract: above ten
// options, labels are inserted as known single tokens rather than allowing BPE
// to merge their textual spelling with the surrounding punctuation.
func (r *Runtime) BuildPromptIDs(state, question string, options []string) ([]int, error) {
	text, err := BuildPrompt(state, question, options)
	if err != nil {
		return nil, err
	}
	if len(options) <= 10 {
		return r.TokenizePrompt(text)
	}
	if r == nil || r.Tokenizer == nil || len(r.labels) != MaxChoice {
		return nil, fmt.Errorf("decider: nil runtime")
	}
	ids := r.Tokenizer.Encode("Context:\n" + state + "\n\nQuestion: " + question + "\nOptions:")
	open := r.Tokenizer.Encode("\n(")
	for i, option := range options {
		ids = append(ids, open...)
		ids = append(ids, r.labels[i])
		ids = append(ids, r.Tokenizer.Encode(") "+option)...)
	}
	ids = append(ids, r.Tokenizer.Encode("\nAnswer: (")...)
	return r.admitIDs(ids)
}

func (r *Runtime) ScoreTokenIDs(ids []int, nopts int) ([]float32, error) {
	if r == nil || r.Bundle == nil || len(ids) == 0 || len(ids) > r.maxInput || nopts < 2 || nopts > MaxChoice {
		return nil, fmt.Errorf("decider: invalid token sequence or option count")
	}
	inputs := make([][]float32, len(ids))
	for i, id := range ids {
		row, err := r.embeddingRow(id)
		if err != nil {
			return nil, err
		}
		inputs[i] = row
	}
	state, err := r.Bundle.NewForwardState()
	if err != nil {
		return nil, err
	}
	outs, _, err := r.Bundle.Base.ForwardChunkLayerStreamed(inputs, state, r.rope, 1e-6, r.Bundle.Meta)
	if err != nil {
		return nil, err
	}
	hidden := outs[len(outs)-1]
	rmsNormZeroCentered(hidden, r.norm, 1e-6)
	roundBF16(hidden)
	logits := make([]float32, nopts)
	for i, id := range r.labels[:nopts] {
		logits[i], err = r.embeddingRowDot(id, hidden)
		if err != nil {
			return nil, err
		}
	}
	return logits, nil
}

func (r *Runtime) ScorePrompt(text string, nopts int) ([]float32, []int, error) {
	ids, err := r.TokenizePrompt(text)
	if err != nil {
		return nil, nil, err
	}
	logits, err := r.ScoreTokenIDs(ids, nopts)
	return logits, ids, err
}

func (r *Runtime) scoreRow(state string, row scoringRow) ([]float32, []int, error) {
	ids, err := r.BuildPromptIDs(state, row.question, row.options)
	if err != nil {
		return nil, nil, err
	}
	logits, err := r.ScoreTokenIDs(ids, len(row.options))
	return logits, ids, err
}

func (r *Runtime) embeddingRow(id int) ([]float32, error) {
	if id < 0 || id >= r.embedding.shape[0] {
		return nil, fmt.Errorf("decider: token %d out of range", id)
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
		return nil, fmt.Errorf("decider: unsupported embedding dtype %s", r.embedding.dtype)
	}
	return out, nil
}

func (r *Runtime) embeddingRowDot(row int, x []float32) (float32, error) {
	if row < 0 || row >= r.embedding.shape[0] || len(x) != r.embedding.shape[1] {
		return 0, fmt.Errorf("decider: invalid tied head row")
	}
	var sum float32
	switch r.embedding.dtype {
	case "BF16":
		off := row * len(x) * 2
		for i, v := range x {
			sum += v * half.BF16ToF32(binary.LittleEndian.Uint16(r.embedding.raw[off+i*2:]))
		}
	case "F32":
		off := row * len(x) * 4
		for i, v := range x {
			sum += v * math.Float32frombits(binary.LittleEndian.Uint32(r.embedding.raw[off+i*4:]))
		}
	default:
		return 0, fmt.Errorf("decider: unsupported embedding dtype %s", r.embedding.dtype)
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

func isFinite(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }
