package nimble

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model/qwen"
)

type Runtime struct {
	Bundle        *qwen.Qwen35NativeMTPBundle
	Tokenizer     *tokenizer.Tokenizer
	embedding, lm rawTensor
	norm, rope    []float32
	maxInput      int
}
type rawTensor struct {
	raw   []byte
	dtype string
	shape []int
}

func LoadMerged(dir string, maxInput int) (*Runtime, error) {
	if maxInput < 1 || maxInput > 2048 {
		return nil, fmt.Errorf("nimble: max input tokens must be 1..2048")
	}
	bundle, err := qwen.LoadQwen35NativeMTPBundleFromDir(dir)
	if err != nil {
		return nil, err
	}
	fail := func(e error) (*Runtime, error) { bundle.Close(); return nil, e }
	src := bundle.TensorSource()
	if src == nil {
		return fail(fmt.Errorf("nimble: missing tensor source"))
	}
	get := func(names ...string) (rawTensor, error) {
		var last error
		for _, name := range names {
			r, d, s, e := src.GetRaw(name)
			if e == nil {
				return rawTensor{r, d, s}, nil
			}
			last = e
		}
		return rawTensor{}, last
	}
	emb, err := get("model.language_model.embed_tokens.weight", "language_model.model.embed_tokens.weight")
	if err != nil {
		return fail(err)
	}
	normRaw, err := src.Get("model.language_model.norm.weight", []int{bundle.Meta.HiddenSize})
	if err != nil {
		return fail(err)
	}
	lm, err := get("lm_head.weight", "language_model.lm_head.weight")
	if err != nil {
		return fail(err)
	}
	tok, err := loadTokenizer(dir)
	if err != nil {
		return fail(err)
	}
	if len(emb.shape) != 2 || emb.shape[0] != bundle.Meta.VocabSize || emb.shape[1] != bundle.Meta.HiddenSize || len(lm.shape) != 2 || lm.shape[0] != bundle.Meta.VocabSize || lm.shape[1] != bundle.Meta.HiddenSize {
		return fail(fmt.Errorf("nimble: embedding/head shape mismatch"))
	}
	ropeMax := bundle.Meta.MaxPositionEmbeddings
	if ropeMax <= 0 || ropeMax > maxInput {
		ropeMax = maxInput
	}
	return &Runtime{Bundle: bundle, Tokenizer: tok, embedding: emb, norm: normRaw.Data(), lm: lm, rope: qwen.NewQwen35RoPEFreqs(bundle.Meta, ropeMax), maxInput: maxInput}, nil
}
func loadTokenizer(dir string) (*tokenizer.Tokenizer, error) {
	tok, err := tokenizer.LoadWithConfig(dir)
	if err == nil {
		if _, a := tok.AddedTokens["<think>"]; a {
			if _, b := tok.AddedTokens["</think>"]; b {
				return tok, nil
			}
		}
		err = fmt.Errorf("nimble: tokenizer sidecar omits ordinary thinking tokens")
	}
	tok, baseErr := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if baseErr != nil {
		return nil, err
	}
	cfgPath := filepath.Join(dir, "tokenizer_config.json")
	raw, readErr := os.ReadFile(cfgPath)
	if readErr != nil {
		return nil, err
	}
	var cfg struct {
		Added map[string]struct {
			Content    string `json:"content"`
			Special    bool   `json:"special"`
			SingleWord bool   `json:"single_word"`
			LStrip     bool   `json:"lstrip"`
			RStrip     bool   `json:"rstrip"`
			Normalized bool   `json:"normalized"`
		} `json:"added_tokens_decoder"`
	}
	if json.Unmarshal(raw, &cfg) != nil || len(cfg.Added) != 0 {
		return nil, err
	}
	for _, content := range []string{"<think>", "</think>"} {
		id, ok := tok.Vocab[content]
		if !ok {
			return nil, err
		}
		if tok.AddedTokens == nil {
			tok.AddedTokens = map[string]int{}
		}
		tok.AddedTokens[content] = id
	}
	return tok, nil
}

func (r *Runtime) Close() error {
	if r == nil || r.Bundle == nil {
		return nil
	}
	return r.Bundle.Close()
}
func (r *Runtime) Tokenize(prompts []Prompt) ([][]int, [][]int, error) {
	if r == nil || r.Tokenizer == nil {
		return nil, nil, fmt.Errorf("nimble: nil runtime")
	}
	ids := make([][]int, len(prompts))
	candidates := make([][]int, len(prompts))
	for i, p := range prompts {
		ids[i] = r.Tokenizer.Encode(p.Text)
		if len(ids[i]) < 1 || len(ids[i]) > r.maxInput {
			return nil, nil, fmt.Errorf("nimble: prompt %q has %d tokens; limit %d", p.Name, len(ids[i]), r.maxInput)
		}
		candidates[i] = make([]int, len(p.Choices))
		for j := range p.Choices {
			code := string(rune('A' + j))
			combined := r.Tokenizer.Encode(p.Text + code)
			if len(combined) != len(ids[i])+1 {
				return nil, nil, fmt.Errorf("nimble: choice code %s is not one token at answer boundary", code)
			}
			for k := range ids[i] {
				if combined[k] != ids[i][k] {
					return nil, nil, fmt.Errorf("nimble: choice code %s changes prompt boundary", code)
				}
			}
			candidates[i][j] = combined[len(ids[i])]
		}
	}
	return ids, candidates, nil
}
func (r *Runtime) Score(context string, fields []Field, temperature float32) (Result, error) {
	prompts, err := RenderPrompts(context, fields)
	if err != nil {
		return Result{}, err
	}
	ids, candidates, err := r.Tokenize(prompts)
	if err != nil {
		return Result{}, err
	}
	rows := make([][]float32, len(ids))
	for i := range ids {
		rows[i], err = r.ScoreTokenIDs(ids[i], candidates[i])
		if err != nil {
			return Result{}, err
		}
	}
	return Summarize(context, "bespokelabs/Bespoke-Nimble-9B", ModelPin, fields, candidates, rows, temperature)
}
func (r *Runtime) forwardTokens(ids []int, state qwen.Qwen35BaseForwardState) (qwen.Qwen35BaseForwardState, []float32, error) {
	if len(ids) == 0 {
		return state, nil, fmt.Errorf("nimble: empty token sequence")
	}
	inputs := make([][]float32, len(ids))
	for i, id := range ids {
		row, err := r.embeddingRow(id)
		if err != nil {
			return state, nil, err
		}
		inputs[i] = row
	}
	outs, next, err := r.Bundle.Base.ForwardChunkLayerStreamed(inputs, state, r.rope, 1e-6, r.Bundle.Meta)
	if err != nil {
		return state, nil, err
	}
	return next, outs[len(outs)-1], nil
}
func (r *Runtime) projectCandidates(hidden []float32, candidates []int) ([]float32, error) {
	if len(hidden) != r.Bundle.Meta.HiddenSize || len(candidates) == 0 {
		return nil, fmt.Errorf("nimble: invalid hidden or candidates")
	}
	rmsNormZeroCentered(hidden, r.norm, 1e-6)
	roundBF16(hidden)
	out := make([]float32, len(candidates))
	for i, id := range candidates {
		value, err := r.lmRowDot(id, hidden)
		if err != nil {
			return nil, err
		}
		out[i] = value
	}
	return out, nil
}
func (r *Runtime) ScoreTokenIDs(ids, candidates []int) ([]float32, error) {
	return r.ScoreTokenIDsSplit(ids, candidates, len(ids))
}
func (r *Runtime) ScoreTokenIDsSplit(ids, candidates []int, split int) ([]float32, error) {
	if r == nil || r.Bundle == nil || len(ids) == 0 || len(ids) > r.maxInput || split < 0 || split > len(ids) {
		return nil, fmt.Errorf("nimble: invalid token sequence")
	}
	state, err := r.Bundle.NewForwardState()
	if err != nil {
		return nil, err
	}
	var hidden []float32
	if split > 0 {
		state, hidden, err = r.forwardTokens(ids[:split], state)
		if err != nil {
			return nil, err
		}
	}
	if split < len(ids) {
		state, hidden, err = r.forwardTokens(ids[split:], state)
		_ = state
		if err != nil {
			return nil, err
		}
	}
	return r.projectCandidates(hidden, candidates)
}
func (r *Runtime) embeddingRow(id int) ([]float32, error) {
	if id < 0 || id >= r.embedding.shape[0] {
		return nil, fmt.Errorf("nimble: token %d out of range", id)
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
		return nil, fmt.Errorf("nimble: unsupported embedding dtype %s", r.embedding.dtype)
	}
	return out, nil
}
func (r *Runtime) lmRowDot(row int, x []float32) (float32, error) {
	if row < 0 || row >= r.lm.shape[0] || len(x) != r.lm.shape[1] {
		return 0, fmt.Errorf("nimble: invalid LM head row")
	}
	var sum float32
	switch r.lm.dtype {
	case "BF16":
		off := row * len(x) * 2
		for i, v := range x {
			sum += v * half.BF16ToF32(binary.LittleEndian.Uint16(r.lm.raw[off+i*2:]))
		}
	case "F32":
		off := row * len(x) * 4
		for i, v := range x {
			sum += v * math.Float32frombits(binary.LittleEndian.Uint32(r.lm.raw[off+i*4:]))
		}
	default:
		return 0, fmt.Errorf("nimble: unsupported LM head dtype %s", r.lm.dtype)
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
func LoadContract(path string) (map[string]any, error) {
	b, e := os.ReadFile(filepath.Clean(path))
	if e != nil {
		return nil, e
	}
	var v map[string]any
	if e = json.Unmarshal(b, &v); e != nil {
		return nil, e
	}
	return v, nil
}
