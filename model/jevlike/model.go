// Copyright (c) 2026 Rui Carmo
// SPDX-License-Identifier: MIT

package jevlike

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

const (
	tinyVocabSize      = 257
	layerNormEpsilon   = float32(1e-5)
	maskedFillValue    = float32(-math.MaxFloat32)
	defaultHeadPrefix  = "head"
	embeddingParamName = "embedding.weight"
	positionParamName  = "position.weight"
)

// Config is the tiny native scorer contract.
type Config struct {
	Width         int `json:"width"`
	Rank          int `json:"rank"`
	ContextTokens int `json:"context_tokens"`
	OptionTokens  int `json:"option_tokens"`
}

// Validate rejects unsupported scorer dimensions.
func (c Config) Validate() error {
	if c.Width <= 0 || c.Rank <= 0 || c.ContextTokens <= 0 || c.OptionTokens < 0 {
		return fmt.Errorf("invalid jevlike scorer config: %+v", c)
	}
	return nil
}

// NamedParameter stores one flattened parameter array and its logical shape.
type NamedParameter struct {
	Name   string    `json:"name"`
	Shape  []int     `json:"shape"`
	Values []float32 `json:"values"`
}

// BatchPrediction is the native batch scoring output.
type BatchPrediction struct {
	Logits        [][]float32 `json:"logits"`
	Probabilities [][]float32 `json:"probabilities"`
}

// AttentionHead mirrors the Python one-pass scorer head.
type AttentionHead struct {
	Width int `json:"width"`
	Rank  int `json:"rank"`

	ContextNormWeight []float32 `json:"context_norm_weight"`
	ContextNormBias   []float32 `json:"context_norm_bias"`
	OptionNormWeight  []float32 `json:"option_norm_weight"`
	OptionNormBias    []float32 `json:"option_norm_bias"`
	QueryWeight       []float32 `json:"query_weight"`
	KeyWeight         []float32 `json:"key_weight"`
	ValueWeight       []float32 `json:"value_weight"`
}

// TinyScorer is the native byte-level scorer.
type TinyScorer struct {
	Config Config `json:"config"`

	EmbeddingWeight []float32     `json:"embedding_weight"`
	PositionWeight  []float32     `json:"position_weight"`
	Head            AttentionHead `json:"head"`
}

// NewAttentionHead allocates one reusable scorer head.
func NewAttentionHead(width, rank int) (*AttentionHead, error) {
	if width <= 0 || rank <= 0 {
		return nil, fmt.Errorf("invalid jevlike attention head width=%d rank=%d", width, rank)
	}
	return &AttentionHead{
		Width:             width,
		Rank:              rank,
		ContextNormWeight: filledFloat32(width, 1),
		ContextNormBias:   make([]float32, width),
		OptionNormWeight:  filledFloat32(width, 1),
		OptionNormBias:    make([]float32, width),
		QueryWeight:       make([]float32, rank*width),
		KeyWeight:         make([]float32, rank*width),
		ValueWeight:       make([]float32, rank*width),
	}, nil
}

// Validate checks parameter lengths.
func (h *AttentionHead) Validate() error {
	if h == nil {
		return fmt.Errorf("jevlike attention head is nil")
	}
	if h.Width <= 0 || h.Rank <= 0 {
		return fmt.Errorf("invalid jevlike attention head dims width=%d rank=%d", h.Width, h.Rank)
	}
	if len(h.ContextNormWeight) != h.Width || len(h.ContextNormBias) != h.Width || len(h.OptionNormWeight) != h.Width || len(h.OptionNormBias) != h.Width {
		return fmt.Errorf("invalid jevlike attention head norm lengths width=%d context_weight=%d context_bias=%d option_weight=%d option_bias=%d", h.Width, len(h.ContextNormWeight), len(h.ContextNormBias), len(h.OptionNormWeight), len(h.OptionNormBias))
	}
	wantProjection := h.Rank * h.Width
	if len(h.QueryWeight) != wantProjection || len(h.KeyWeight) != wantProjection || len(h.ValueWeight) != wantProjection {
		return fmt.Errorf("invalid jevlike attention head projection lengths want=%d query=%d key=%d value=%d", wantProjection, len(h.QueryWeight), len(h.KeyWeight), len(h.ValueWeight))
	}
	return nil
}

// NamedParameters returns flattened arrays named like the Python state_dict.
func (h *AttentionHead) NamedParameters(prefix string) []NamedParameter {
	if prefix == "" {
		prefix = defaultHeadPrefix
	}
	return []NamedParameter{
		{Name: prefix + ".context_norm.weight", Shape: []int{h.Width}, Values: append([]float32(nil), h.ContextNormWeight...)},
		{Name: prefix + ".context_norm.bias", Shape: []int{h.Width}, Values: append([]float32(nil), h.ContextNormBias...)},
		{Name: prefix + ".option_norm.weight", Shape: []int{h.Width}, Values: append([]float32(nil), h.OptionNormWeight...)},
		{Name: prefix + ".option_norm.bias", Shape: []int{h.Width}, Values: append([]float32(nil), h.OptionNormBias...)},
		{Name: prefix + ".query.weight", Shape: []int{h.Rank, h.Width}, Values: append([]float32(nil), h.QueryWeight...)},
		{Name: prefix + ".key.weight", Shape: []int{h.Rank, h.Width}, Values: append([]float32(nil), h.KeyWeight...)},
		{Name: prefix + ".value.weight", Shape: []int{h.Rank, h.Width}, Values: append([]float32(nil), h.ValueWeight...)},
	}
}

// NamedParameterMap returns named flattened arrays keyed like the Python state_dict.
func (h *AttentionHead) NamedParameterMap(prefix string) map[string][]float32 {
	params := h.NamedParameters(prefix)
	out := make(map[string][]float32, len(params))
	for _, param := range params {
		out[param.Name] = append([]float32(nil), param.Values...)
	}
	return out
}

// LoadNamedParameters copies the provided named arrays into the head.
func (h *AttentionHead) LoadNamedParameters(prefix string, params map[string][]float32) error {
	if err := h.Validate(); err != nil {
		return err
	}
	if prefix == "" {
		prefix = defaultHeadPrefix
	}
	specs := map[string]*[]float32{
		prefix + ".context_norm.weight": &h.ContextNormWeight,
		prefix + ".context_norm.bias":   &h.ContextNormBias,
		prefix + ".option_norm.weight":  &h.OptionNormWeight,
		prefix + ".option_norm.bias":    &h.OptionNormBias,
		prefix + ".query.weight":        &h.QueryWeight,
		prefix + ".key.weight":          &h.KeyWeight,
		prefix + ".value.weight":        &h.ValueWeight,
	}
	for name, values := range params {
		dst, ok := specs[name]
		if !ok {
			return fmt.Errorf("unknown jevlike attention parameter %q", name)
		}
		if len(values) != len(*dst) {
			return fmt.Errorf("jevlike attention parameter %q length=%d, want %d", name, len(values), len(*dst))
		}
		copy(*dst, values)
	}
	return nil
}

// Forward scores pooled options against context tokens.
func (h *AttentionHead) Forward(context [][][]float32, contextMask [][]bool, options [][][]float32, optionMask [][]bool) ([][]float32, error) {
	if err := h.Validate(); err != nil {
		return nil, err
	}
	rows, contextTokens, optionsPerRow, err := validateHeadInputs(context, contextMask, options, optionMask, h.Width)
	if err != nil {
		return nil, err
	}
	logits := make([][]float32, rows)
	rankScale := float32(math.Sqrt(float64(h.Rank)))
	for row := 0; row < rows; row++ {
		keys := make([][]float32, contextTokens)
		values := make([][]float32, contextTokens)
		for token := 0; token < contextTokens; token++ {
			normalized := layerNormVector(context[row][token], h.ContextNormWeight, h.ContextNormBias)
			keys[token] = linearNoBias(h.KeyWeight, h.Rank, h.Width, normalized)
			values[token] = linearNoBias(h.ValueWeight, h.Rank, h.Width, normalized)
		}
		logits[row] = make([]float32, optionsPerRow)
		for option := 0; option < optionsPerRow; option++ {
			normalized := layerNormVector(options[row][option], h.OptionNormWeight, h.OptionNormBias)
			query := linearNoBias(h.QueryWeight, h.Rank, h.Width, normalized)
			scores := make([]float32, contextTokens)
			for token := 0; token < contextTokens; token++ {
				scores[token] = dotFloat32(query, keys[token]) / rankScale
				if !contextMask[row][token] {
					scores[token] = maskedFillValue
				}
			}
			attention := softmaxFloat32(scores)
			attended := make([]float32, h.Rank)
			for token, weight := range attention {
				for dim := 0; dim < h.Rank; dim++ {
					attended[dim] += weight * values[token][dim]
				}
			}
			logits[row][option] = dotFloat32(query, attended) / rankScale
			if !optionMask[row][option] {
				logits[row][option] = maskedFillValue
			}
		}
	}
	return logits, nil
}

// NewTinyScorer allocates a zero-initialized scorer with identity layer norms.
func NewTinyScorer(config Config) (*TinyScorer, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	head, err := NewAttentionHead(config.Width, config.Rank)
	if err != nil {
		return nil, err
	}
	return &TinyScorer{
		Config:          config,
		EmbeddingWeight: make([]float32, tinyVocabSize*config.Width),
		PositionWeight:  make([]float32, config.ContextTokens*config.Width),
		Head:            *head,
	}, nil
}

// Validate checks scorer dimensions and parameter lengths.
func (m *TinyScorer) Validate() error {
	if m == nil {
		return fmt.Errorf("jevlike scorer is nil")
	}
	if err := m.Config.Validate(); err != nil {
		return err
	}
	if len(m.EmbeddingWeight) != tinyVocabSize*m.Config.Width {
		return fmt.Errorf("invalid jevlike embedding length=%d, want %d", len(m.EmbeddingWeight), tinyVocabSize*m.Config.Width)
	}
	if len(m.PositionWeight) != m.Config.ContextTokens*m.Config.Width {
		return fmt.Errorf("invalid jevlike position length=%d, want %d", len(m.PositionWeight), m.Config.ContextTokens*m.Config.Width)
	}
	if m.Head.Width != m.Config.Width || m.Head.Rank != m.Config.Rank {
		return fmt.Errorf("jevlike scorer/head mismatch config=%+v head_width=%d head_rank=%d", m.Config, m.Head.Width, m.Head.Rank)
	}
	return m.Head.Validate()
}

// NamedParameters returns flattened arrays named like the Python state_dict.
func (m *TinyScorer) NamedParameters() []NamedParameter {
	params := []NamedParameter{
		{Name: embeddingParamName, Shape: []int{tinyVocabSize, m.Config.Width}, Values: append([]float32(nil), m.EmbeddingWeight...)},
		{Name: positionParamName, Shape: []int{m.Config.ContextTokens, m.Config.Width}, Values: append([]float32(nil), m.PositionWeight...)},
	}
	return append(params, m.Head.NamedParameters(defaultHeadPrefix)...)
}

// NamedParameterMap returns named flattened arrays keyed like the Python state_dict.
func (m *TinyScorer) NamedParameterMap() map[string][]float32 {
	params := m.NamedParameters()
	out := make(map[string][]float32, len(params))
	for _, param := range params {
		out[param.Name] = append([]float32(nil), param.Values...)
	}
	return out
}

// LoadNamedParameters copies provided named arrays into the scorer.
func (m *TinyScorer) LoadNamedParameters(params map[string][]float32) error {
	if err := m.Validate(); err != nil {
		return err
	}
	headParams := make(map[string][]float32)
	for name, values := range params {
		switch name {
		case embeddingParamName:
			if len(values) != len(m.EmbeddingWeight) {
				return fmt.Errorf("jevlike parameter %q length=%d, want %d", name, len(values), len(m.EmbeddingWeight))
			}
			copy(m.EmbeddingWeight, values)
		case positionParamName:
			if len(values) != len(m.PositionWeight) {
				return fmt.Errorf("jevlike parameter %q length=%d, want %d", name, len(values), len(m.PositionWeight))
			}
			copy(m.PositionWeight, values)
		default:
			headParams[name] = values
		}
	}
	return m.Head.LoadNamedParameters(defaultHeadPrefix, headParams)
}

// Forward returns one masked logit per option.
func (m *TinyScorer) Forward(batch ByteBatch) ([][]float32, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	rows, contextTokens, optionsPerRow, optionTokens, err := validateBatchForScorer(batch, m.Config)
	if err != nil {
		return nil, err
	}
	context := make([][][]float32, rows)
	options := make([][][]float32, rows)
	for row := 0; row < rows; row++ {
		context[row] = make([][]float32, contextTokens)
		for token := 0; token < contextTokens; token++ {
			vector := make([]float32, m.Config.Width)
			copy(vector, m.embeddingRow(int(batch.ContextIDs[row][token])))
			position := m.positionRow(token)
			for dim := 0; dim < m.Config.Width; dim++ {
				vector[dim] += position[dim]
			}
			context[row][token] = vector
		}
		options[row] = make([][]float32, optionsPerRow)
		for option := 0; option < optionsPerRow; option++ {
			vector := make([]float32, m.Config.Width)
			count := 0
			for token := 0; token < optionTokens; token++ {
				if !batch.OptionTokenMask[row][option][token] {
					continue
				}
				embedding := m.embeddingRow(int(batch.OptionIDs[row][option][token]))
				for dim := 0; dim < m.Config.Width; dim++ {
					vector[dim] += embedding[dim]
				}
				count++
			}
			if count > 1 {
				denominator := float32(count)
				for dim := 0; dim < m.Config.Width; dim++ {
					vector[dim] /= denominator
				}
			}
			options[row][option] = vector
		}
	}
	return m.Head.Forward(context, batch.ContextMask, options, batch.OptionMask)
}

// PredictBatch returns masked logits and option probabilities.
func (m *TinyScorer) PredictBatch(batch ByteBatch) (BatchPrediction, error) {
	logits, err := m.Forward(batch)
	if err != nil {
		return BatchPrediction{}, err
	}
	probabilities := make([][]float32, len(logits))
	for row := range logits {
		probabilities[row] = softmaxFloat32(logits[row])
	}
	return BatchPrediction{Logits: logits, Probabilities: probabilities}, nil
}

// Predict is a convenience alias for PredictBatch.
func (m *TinyScorer) Predict(batch ByteBatch) (BatchPrediction, error) {
	return m.PredictBatch(batch)
}

func (m *TinyScorer) embeddingRow(id int) []float32 {
	start := id * m.Config.Width
	return m.EmbeddingWeight[start : start+m.Config.Width]
}

func (m *TinyScorer) positionRow(position int) []float32 {
	start := position * m.Config.Width
	return m.PositionWeight[start : start+m.Config.Width]
}

func validateHeadInputs(context [][][]float32, contextMask [][]bool, options [][][]float32, optionMask [][]bool, width int) (rows, contextTokens, optionsPerRow int, err error) {
	rows = len(context)
	if rows == 0 {
		return 0, 0, 0, fmt.Errorf("jevlike attention inputs need at least one row")
	}
	if len(contextMask) != rows || len(options) != rows || len(optionMask) != rows {
		return 0, 0, 0, fmt.Errorf("jevlike attention row mismatch context=%d context_mask=%d options=%d option_mask=%d", len(context), len(contextMask), len(options), len(optionMask))
	}
	contextTokens = len(context[0])
	optionsPerRow = len(options[0])
	if optionsPerRow == 0 {
		return 0, 0, 0, fmt.Errorf("jevlike attention inputs need at least one option")
	}
	for row := 0; row < rows; row++ {
		if len(context[row]) != contextTokens || len(contextMask[row]) != contextTokens {
			return 0, 0, 0, fmt.Errorf("jevlike attention context shape mismatch row=%d tokens=%d mask=%d want=%d", row, len(context[row]), len(contextMask[row]), contextTokens)
		}
		if len(options[row]) != optionsPerRow || len(optionMask[row]) != optionsPerRow {
			return 0, 0, 0, fmt.Errorf("jevlike attention option shape mismatch row=%d options=%d mask=%d want=%d", row, len(options[row]), len(optionMask[row]), optionsPerRow)
		}
		for token := 0; token < contextTokens; token++ {
			if len(context[row][token]) != width {
				return 0, 0, 0, fmt.Errorf("jevlike attention context width mismatch row=%d token=%d width=%d want=%d", row, token, len(context[row][token]), width)
			}
		}
		present := 0
		for option := 0; option < optionsPerRow; option++ {
			if len(options[row][option]) != width {
				return 0, 0, 0, fmt.Errorf("jevlike attention option width mismatch row=%d option=%d width=%d want=%d", row, option, len(options[row][option]), width)
			}
			if optionMask[row][option] {
				present++
			}
		}
		if present == 0 {
			return 0, 0, 0, fmt.Errorf("jevlike attention row=%d has no active options", row)
		}
	}
	return rows, contextTokens, optionsPerRow, nil
}

func validateBatchForScorer(batch ByteBatch, config Config) (rows, contextTokens, optionsPerRow, optionTokens int, err error) {
	rows = len(batch.ContextIDs)
	if rows == 0 {
		return 0, 0, 0, 0, fmt.Errorf("jevlike scorer batch needs at least one example")
	}
	if len(batch.ContextMask) != rows || len(batch.OptionIDs) != rows || len(batch.OptionTokenMask) != rows || len(batch.OptionMask) != rows {
		return 0, 0, 0, 0, fmt.Errorf("jevlike scorer batch row mismatch context=%d context_mask=%d options=%d option_token_mask=%d option_mask=%d", len(batch.ContextIDs), len(batch.ContextMask), len(batch.OptionIDs), len(batch.OptionTokenMask), len(batch.OptionMask))
	}
	if len(batch.Labels) != 0 && len(batch.Labels) != rows {
		return 0, 0, 0, 0, fmt.Errorf("jevlike scorer labels=%d, want 0 or %d", len(batch.Labels), rows)
	}
	contextTokens = len(batch.ContextIDs[0])
	optionsPerRow = len(batch.OptionIDs[0])
	if optionsPerRow == 0 {
		return 0, 0, 0, 0, fmt.Errorf("jevlike scorer batch needs at least one option")
	}
	optionTokens = 0
	if optionsPerRow > 0 {
		optionTokens = len(batch.OptionIDs[0][0])
	}
	if contextTokens > config.ContextTokens {
		return 0, 0, 0, 0, fmt.Errorf("jevlike scorer context tokens=%d exceed config limit=%d", contextTokens, config.ContextTokens)
	}
	if optionTokens > config.OptionTokens {
		return 0, 0, 0, 0, fmt.Errorf("jevlike scorer option tokens=%d exceed config limit=%d", optionTokens, config.OptionTokens)
	}
	for row := 0; row < rows; row++ {
		if len(batch.ContextIDs[row]) != contextTokens || len(batch.ContextMask[row]) != contextTokens {
			return 0, 0, 0, 0, fmt.Errorf("jevlike scorer context shape mismatch row=%d ids=%d mask=%d want=%d", row, len(batch.ContextIDs[row]), len(batch.ContextMask[row]), contextTokens)
		}
		if len(batch.OptionIDs[row]) != optionsPerRow || len(batch.OptionTokenMask[row]) != optionsPerRow || len(batch.OptionMask[row]) != optionsPerRow {
			return 0, 0, 0, 0, fmt.Errorf("jevlike scorer option shape mismatch row=%d ids=%d token_mask=%d option_mask=%d want=%d", row, len(batch.OptionIDs[row]), len(batch.OptionTokenMask[row]), len(batch.OptionMask[row]), optionsPerRow)
		}
		present := 0
		for token, id := range batch.ContextIDs[row] {
			if id >= tinyVocabSize {
				return 0, 0, 0, 0, fmt.Errorf("jevlike scorer context id row=%d token=%d id=%d out of range", row, token, id)
			}
		}
		for option := 0; option < optionsPerRow; option++ {
			if len(batch.OptionIDs[row][option]) != optionTokens || len(batch.OptionTokenMask[row][option]) != optionTokens {
				return 0, 0, 0, 0, fmt.Errorf("jevlike scorer option token shape mismatch row=%d option=%d ids=%d mask=%d want=%d", row, option, len(batch.OptionIDs[row][option]), len(batch.OptionTokenMask[row][option]), optionTokens)
			}
			if batch.OptionMask[row][option] {
				present++
			}
			for token, id := range batch.OptionIDs[row][option] {
				if id >= tinyVocabSize {
					return 0, 0, 0, 0, fmt.Errorf("jevlike scorer option id row=%d option=%d token=%d id=%d out of range", row, option, token, id)
				}
				if !batch.OptionMask[row][option] && batch.OptionTokenMask[row][option][token] {
					return 0, 0, 0, 0, fmt.Errorf("jevlike scorer option row=%d option=%d has token mask set while option is masked", row, option)
				}
			}
		}
		if present == 0 {
			return 0, 0, 0, 0, fmt.Errorf("jevlike scorer row=%d has no active options", row)
		}
		if len(batch.Labels) == rows {
			if batch.Labels[row] < 0 || batch.Labels[row] >= optionsPerRow || !batch.OptionMask[row][batch.Labels[row]] {
				return 0, 0, 0, 0, fmt.Errorf("jevlike scorer label row=%d label=%d out of range or masked", row, batch.Labels[row])
			}
		}
	}
	return rows, contextTokens, optionsPerRow, optionTokens, nil
}

func filledFloat32(n int, value float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = value
	}
	return out
}

func layerNormVector(input, gamma, beta []float32) []float32 {
	out := make([]float32, len(input))
	if len(input) == 0 {
		return out
	}
	var mean float64
	for _, value := range input {
		mean += float64(value)
	}
	mean /= float64(len(input))
	var variance float64
	for _, value := range input {
		delta := float64(value) - mean
		variance += delta * delta
	}
	variance /= float64(len(input))
	invStd := 1 / math.Sqrt(variance+float64(layerNormEpsilon))
	for i, value := range input {
		normalized := (float64(value) - mean) * invStd
		if len(gamma) != 0 {
			normalized *= float64(gamma[i])
		}
		if len(beta) != 0 {
			normalized += float64(beta[i])
		}
		out[i] = float32(normalized)
	}
	return out
}

func linearNoBias(weight []float32, outDim, inDim int, input []float32) []float32 {
	out := make([]float32, outDim)
	// GemvRows dispatches to native Plan 9 kernels on supported CPUs and
	// retains the runtime's portable implementation elsewhere.
	if simd.GemvRows(out, input, weight, outDim, inDim) {
		return out
	}
	for row := 0; row < outDim; row++ {
		base := row * inDim
		var sum float64
		for col, value := range input {
			sum += float64(weight[base+col]) * float64(value)
		}
		out[row] = float32(sum)
	}
	return out
}

func dotFloat32(a, b []float32) float32 {
	return simd.Sdot(a, b)
}

func softmaxFloat32(values []float32) []float32 {
	if len(values) == 0 {
		return nil
	}
	maxValue := values[0]
	for _, value := range values[1:] {
		if value > maxValue {
			maxValue = value
		}
	}
	out := make([]float32, len(values))
	var sum float64
	for i, value := range values {
		expValue := math.Exp(float64(value - maxValue))
		out[i] = float32(expValue)
		sum += expValue
	}
	if sum == 0 {
		return out
	}
	inv := float32(1 / sum)
	for i := range out {
		out[i] *= inv
	}
	return out
}
