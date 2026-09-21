// Copyright (c) 2026 Rui Carmo
// SPDX-License-Identifier: MIT

package jevlike

import (
	"fmt"
	"math"
)

// RGBFrame stores one contiguous HWC RGB frame.
type RGBFrame struct {
	Width  int
	Height int
	Pixels []uint8
}

// Validate rejects malformed RGB frames.
func (f RGBFrame) Validate() error {
	if f.Width <= 0 || f.Height <= 0 {
		return fmt.Errorf("invalid RGB frame size %dx%d", f.Width, f.Height)
	}
	if len(f.Pixels) != f.Width*f.Height*3 {
		return fmt.Errorf("invalid RGB frame pixel length=%d, want %d", len(f.Pixels), f.Width*f.Height*3)
	}
	return nil
}

// PackedObservation stores RGB plus one signed greyscale difference channel.
type PackedObservation struct {
	Width  int
	Height int
	Pixels []uint8
}

// Validate rejects malformed packed observations.
func (o PackedObservation) Validate() error {
	if o.Width <= 0 || o.Height <= 0 {
		return fmt.Errorf("invalid packed observation size %dx%d", o.Width, o.Height)
	}
	if len(o.Pixels) != o.Width*o.Height*4 {
		return fmt.Errorf("invalid packed observation pixel length=%d, want %d", len(o.Pixels), o.Width*o.Height*4)
	}
	return nil
}

// ObservationTensorResult stores one contiguous NCHW batch scaled to [0, 1].
type ObservationTensorResult struct {
	Values []float32
	Shape  [4]int
}

// Observation packs RGB plus a signed greyscale frame-difference channel.
// The caller supplies raw RGB frames; no visual encoder or environment-specific
// stem is implied here.
func Observation(frame RGBFrame, previous *RGBFrame) (PackedObservation, error) {
	if err := frame.Validate(); err != nil {
		return PackedObservation{}, err
	}
	if previous != nil {
		if err := previous.Validate(); err != nil {
			return PackedObservation{}, err
		}
		if previous.Width != frame.Width || previous.Height != frame.Height {
			return PackedObservation{}, fmt.Errorf("previous RGB frame size %dx%d does not match current %dx%d", previous.Width, previous.Height, frame.Width, frame.Height)
		}
	}
	pixels := frame.Width * frame.Height
	out := make([]uint8, pixels*4)
	for i := 0; i < pixels; i++ {
		src := i * 3
		dst := i * 4
		copy(out[dst:dst+3], frame.Pixels[src:src+3])
		if previous == nil {
			out[dst+3] = 128
			continue
		}
		diff := (float64(int(frame.Pixels[src])-int(previous.Pixels[src])) +
			float64(int(frame.Pixels[src+1])-int(previous.Pixels[src+1])) +
			float64(int(frame.Pixels[src+2])-int(previous.Pixels[src+2]))) / 3.0
		out[dst+3] = encodeSignedDifference(diff)
	}
	return PackedObservation{Width: frame.Width, Height: frame.Height, Pixels: out}, nil
}

// ObservationTensor converts packed observations to contiguous NCHW float32
// scaled by 1/255.
func ObservationTensor(items []PackedObservation) (ObservationTensorResult, error) {
	if len(items) == 0 {
		return ObservationTensorResult{}, fmt.Errorf("observation tensor needs at least one packed observation")
	}
	for i := range items {
		if err := items[i].Validate(); err != nil {
			return ObservationTensorResult{}, fmt.Errorf("packed observation %d: %w", i, err)
		}
	}
	width := items[0].Width
	height := items[0].Height
	for i := 1; i < len(items); i++ {
		if items[i].Width != width || items[i].Height != height {
			return ObservationTensorResult{}, fmt.Errorf("packed observation %d size %dx%d does not match batch reference %dx%d", i, items[i].Width, items[i].Height, width, height)
		}
	}
	pixels := width * height
	values := make([]float32, len(items)*4*pixels)
	for b := range items {
		base := b * 4 * pixels
		for i := 0; i < pixels; i++ {
			src := i * 4
			for channel := 0; channel < 4; channel++ {
				values[base+channel*pixels+i] = float32(items[b].Pixels[src+channel]) / 255.0
			}
		}
	}
	return ObservationTensorResult{Values: values, Shape: [4]int{len(items), 4, height, width}}, nil
}

// Position2D returns fixed separable 2D sinusoidal positions with shape
// (rows*columns, width).
func Position2D(rows, columns, width int) ([]float32, error) {
	if rows <= 0 || columns <= 0 || width <= 0 {
		return nil, fmt.Errorf("invalid 2D position shape rows=%d columns=%d width=%d", rows, columns, width)
	}
	if width%4 != 0 {
		return nil, fmt.Errorf("position width=%d must be divisible by four", width)
	}
	quarter := width / 4
	denominator := max(1, quarter-1)
	frequency := make([]float64, quarter)
	for i := range frequency {
		frequency[i] = math.Exp(-math.Log(10_000) * float64(i) / float64(denominator))
	}
	out := make([]float32, rows*columns*width)
	for y := 0; y < rows; y++ {
		for x := 0; x < columns; x++ {
			base := (y*columns + x) * width
			for i, freq := range frequency {
				yScaled := float64(y) * freq
				xScaled := float64(x) * freq
				out[base+i] = float32(math.Sin(yScaled))
				out[base+quarter+i] = float32(math.Cos(yScaled))
				out[base+2*quarter+i] = float32(math.Sin(xScaled))
				out[base+3*quarter+i] = float32(math.Cos(xScaled))
			}
		}
	}
	return out, nil
}

// VisionActionConfig describes a reusable scorer over caller-supplied visual
// feature tokens. It does not include any environment-specific visual stem.
type VisionActionConfig struct {
	Width        int
	Rank         int
	Actions      int
	Reads        int
	PatchRows    int
	PatchColumns int
}

// Validate rejects unsupported scorer dimensions.
func (c VisionActionConfig) Validate() error {
	if c.Width <= 0 || c.Rank <= 0 || c.Actions <= 0 || c.Reads <= 0 || c.PatchRows <= 0 || c.PatchColumns <= 0 {
		return fmt.Errorf("invalid vision action scorer config: %+v", c)
	}
	if c.Width != c.Rank {
		return fmt.Errorf("vision action scorer requires width == rank so fixed positions can be added after key projection: width=%d rank=%d", c.Width, c.Rank)
	}
	return nil
}

// PatchTokens returns the expected number of feature tokens.
func (c VisionActionConfig) PatchTokens() int {
	return c.PatchRows * c.PatchColumns
}

// OptionSelection chooses which learned action embeddings to score.
// Zero value means all configured actions for every batch row.
type OptionSelection struct {
	Shared   []int
	PerBatch [][]int
}

// VisionActionTrace stores averaged read diagnostics.
type VisionActionTrace struct {
	QueryMatrix   [][][]float32 `json:"query_matrix"`
	KeyMatrix     [][][]float32 `json:"key_matrix"`
	ValueMatrix   [][][]float32 `json:"value_matrix"`
	AttentionMap  [][][]float32 `json:"attention_map"`
	LogitsMatrix  [][]float32   `json:"logits_matrix"`
	Probabilities [][]float32   `json:"probabilities"`
	Entropy       [][]float32   `json:"entropy"`
}

// VisionActionScorer applies one or more reusable attention reads over
// caller-supplied patch feature tokens.
type VisionActionScorer struct {
	Config VisionActionConfig

	Positions       []float32
	OptionEmbedding []float32
	Heads           []AttentionHead

	ValueNormWeight []float32
	ValueNormBias   []float32
	ValueWeight     []float32
	ValueBias       float32
}

// NewVisionActionScorer allocates a scorer with identity layer norms and fixed
// 2D positions.
func NewVisionActionScorer(config VisionActionConfig) (*VisionActionScorer, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	positions, err := Position2D(config.PatchRows, config.PatchColumns, config.Rank)
	if err != nil {
		return nil, err
	}
	heads := make([]AttentionHead, config.Reads)
	for i := range heads {
		head, err := NewAttentionHead(config.Width, config.Rank)
		if err != nil {
			return nil, err
		}
		heads[i] = *head
	}
	return &VisionActionScorer{
		Config:          config,
		Positions:       positions,
		OptionEmbedding: make([]float32, config.Actions*config.Width),
		Heads:           heads,
		ValueNormWeight: filledFloat32(config.Width, 1),
		ValueNormBias:   make([]float32, config.Width),
		ValueWeight:     make([]float32, config.Width),
	}, nil
}

// Validate checks scorer dimensions and parameter lengths.
func (m *VisionActionScorer) Validate() error {
	if m == nil {
		return fmt.Errorf("vision action scorer is nil")
	}
	if err := m.Config.Validate(); err != nil {
		return err
	}
	if len(m.Positions) != m.Config.PatchTokens()*m.Config.Rank {
		return fmt.Errorf("invalid vision position length=%d, want %d", len(m.Positions), m.Config.PatchTokens()*m.Config.Rank)
	}
	if len(m.OptionEmbedding) != m.Config.Actions*m.Config.Width {
		return fmt.Errorf("invalid vision option embedding length=%d, want %d", len(m.OptionEmbedding), m.Config.Actions*m.Config.Width)
	}
	if len(m.Heads) != m.Config.Reads {
		return fmt.Errorf("invalid vision head count=%d, want %d", len(m.Heads), m.Config.Reads)
	}
	if len(m.ValueNormWeight) != m.Config.Width || len(m.ValueNormBias) != m.Config.Width || len(m.ValueWeight) != m.Config.Width {
		return fmt.Errorf("invalid value head lengths norm_weight=%d norm_bias=%d weight=%d want=%d", len(m.ValueNormWeight), len(m.ValueNormBias), len(m.ValueWeight), m.Config.Width)
	}
	for i := range m.Heads {
		if m.Heads[i].Width != m.Config.Width || m.Heads[i].Rank != m.Config.Rank {
			return fmt.Errorf("vision head %d dims width=%d rank=%d want width=%d rank=%d", i, m.Heads[i].Width, m.Heads[i].Rank, m.Config.Width, m.Config.Rank)
		}
		if err := m.Heads[i].Validate(); err != nil {
			return fmt.Errorf("vision head %d: %w", i, err)
		}
	}
	return nil
}

// Forward scores selected actions for each batch row of caller-supplied feature
// tokens and returns logits plus one scalar value estimate per row.
func (m *VisionActionScorer) Forward(features [][][]float32, selection OptionSelection) ([][]float32, []float32, error) {
	logits, value, _, err := m.forward(features, selection, false)
	return logits, value, err
}

// ForwardTrace returns logits, values, and averaged attention diagnostics.
func (m *VisionActionScorer) ForwardTrace(features [][][]float32, selection OptionSelection) ([][]float32, []float32, VisionActionTrace, error) {
	return m.forward(features, selection, true)
}

func (m *VisionActionScorer) forward(features [][][]float32, selection OptionSelection, capture bool) ([][]float32, []float32, VisionActionTrace, error) {
	if err := m.Validate(); err != nil {
		return nil, nil, VisionActionTrace{}, err
	}
	rows, tokens, err := validateVisionFeatures(features, m.Config)
	if err != nil {
		return nil, nil, VisionActionTrace{}, err
	}
	optionIDs, err := validateOptionSelection(selection, rows, m.Config.Actions)
	if err != nil {
		return nil, nil, VisionActionTrace{}, err
	}
	optionsPerRow := len(optionIDs[0])
	logits := make2DFloat32(rows, optionsPerRow)
	trace := VisionActionTrace{}
	if capture {
		trace.QueryMatrix = make3DFloat32(rows, optionsPerRow, m.Config.Rank)
		trace.KeyMatrix = make3DFloat32(rows, tokens, m.Config.Rank)
		trace.ValueMatrix = make3DFloat32(rows, tokens, m.Config.Rank)
		trace.AttentionMap = make3DFloat32(rows, optionsPerRow, tokens)
	}
	readScale := float32(1.0 / float64(len(m.Heads)))
	rankScale := float32(math.Sqrt(float64(m.Config.Rank)))
	for read := range m.Heads {
		head := &m.Heads[read]
		for row := 0; row < rows; row++ {
			keys := make([][]float32, tokens)
			values := make([][]float32, tokens)
			for token := 0; token < tokens; token++ {
				normalized := layerNormVector(features[row][token], head.ContextNormWeight, head.ContextNormBias)
				key := linearNoBias(head.KeyWeight, head.Rank, head.Width, normalized)
				position := m.positionRow(token)
				for dim := 0; dim < head.Rank; dim++ {
					key[dim] += position[dim]
				}
				value := linearNoBias(head.ValueWeight, head.Rank, head.Width, normalized)
				keys[token] = key
				values[token] = value
				if capture {
					for dim := 0; dim < head.Rank; dim++ {
						trace.KeyMatrix[row][token][dim] += key[dim] * readScale
						trace.ValueMatrix[row][token][dim] += value[dim] * readScale
					}
				}
			}
			for option := 0; option < optionsPerRow; option++ {
				embedding := m.optionRow(optionIDs[row][option])
				normalized := layerNormVector(embedding, head.OptionNormWeight, head.OptionNormBias)
				query := linearNoBias(head.QueryWeight, head.Rank, head.Width, normalized)
				scores := make([]float32, tokens)
				for token := 0; token < tokens; token++ {
					scores[token] = dotFloat32(query, keys[token]) / rankScale
				}
				attention := softmaxFloat32(scores)
				attended := make([]float32, head.Rank)
				for token, weight := range attention {
					for dim := 0; dim < head.Rank; dim++ {
						attended[dim] += weight * values[token][dim]
					}
				}
				logits[row][option] += dotFloat32(query, attended) / rankScale * readScale
				if capture {
					for dim := 0; dim < head.Rank; dim++ {
						trace.QueryMatrix[row][option][dim] += query[dim] * readScale
					}
					for token := 0; token < tokens; token++ {
						trace.AttentionMap[row][option][token] += attention[token] * readScale
					}
				}
			}
		}
	}
	value := make([]float32, rows)
	for row := 0; row < rows; row++ {
		pooled := make([]float32, m.Config.Width)
		for token := 0; token < tokens; token++ {
			normalized := layerNormVector(features[row][token], m.Heads[0].ContextNormWeight, m.Heads[0].ContextNormBias)
			for dim := 0; dim < m.Config.Width; dim++ {
				pooled[dim] += normalized[dim]
			}
		}
		invTokens := float32(1.0 / float64(tokens))
		for dim := 0; dim < m.Config.Width; dim++ {
			pooled[dim] *= invTokens
		}
		pooled = layerNormVector(pooled, m.ValueNormWeight, m.ValueNormBias)
		value[row] = dotFloat32(pooled, m.ValueWeight) + m.ValueBias
	}
	if !capture {
		return logits, value, VisionActionTrace{}, nil
	}
	trace.LogitsMatrix = clone2DFloat32(logits)
	trace.Probabilities = make2DFloat32(rows, optionsPerRow)
	trace.Entropy = make2DFloat32(rows, optionsPerRow)
	for row := 0; row < rows; row++ {
		trace.Probabilities[row] = softmaxFloat32(logits[row])
		for option := 0; option < optionsPerRow; option++ {
			trace.Entropy[row][option] = normalizedAttentionEntropy(trace.AttentionMap[row][option])
		}
	}
	return logits, value, trace, nil
}

func validateVisionFeatures(features [][][]float32, config VisionActionConfig) (rows, tokens int, err error) {
	rows = len(features)
	if rows == 0 {
		return 0, 0, fmt.Errorf("vision action scorer needs at least one feature row")
	}
	tokens = len(features[0])
	if tokens != config.PatchTokens() {
		return 0, 0, fmt.Errorf("vision action scorer tokens=%d, want %d", tokens, config.PatchTokens())
	}
	for row := 0; row < rows; row++ {
		if len(features[row]) != tokens {
			return 0, 0, fmt.Errorf("vision feature token count row=%d tokens=%d want=%d", row, len(features[row]), tokens)
		}
		for token := 0; token < tokens; token++ {
			if len(features[row][token]) != config.Width {
				return 0, 0, fmt.Errorf("vision feature width row=%d token=%d width=%d want=%d", row, token, len(features[row][token]), config.Width)
			}
		}
	}
	return rows, tokens, nil
}

func validateOptionSelection(selection OptionSelection, rows, actions int) ([][]int, error) {
	if len(selection.Shared) != 0 && len(selection.PerBatch) != 0 {
		return nil, fmt.Errorf("option selection must use shared ids or per-batch ids, not both")
	}
	if len(selection.Shared) == 0 && len(selection.PerBatch) == 0 {
		ids := make([][]int, rows)
		for row := 0; row < rows; row++ {
			ids[row] = make([]int, actions)
			for option := 0; option < actions; option++ {
				ids[row][option] = option
			}
		}
		return ids, nil
	}
	if len(selection.Shared) != 0 {
		if err := validateOptionIDs(selection.Shared, actions, "shared"); err != nil {
			return nil, err
		}
		ids := make([][]int, rows)
		for row := 0; row < rows; row++ {
			ids[row] = append([]int(nil), selection.Shared...)
		}
		return ids, nil
	}
	if len(selection.PerBatch) != rows {
		return nil, fmt.Errorf("per-batch option rows=%d, want %d", len(selection.PerBatch), rows)
	}
	optionsPerRow := len(selection.PerBatch[0])
	if optionsPerRow == 0 {
		return nil, fmt.Errorf("per-batch option selection needs at least one id")
	}
	ids := make([][]int, rows)
	for row := 0; row < rows; row++ {
		if len(selection.PerBatch[row]) != optionsPerRow {
			return nil, fmt.Errorf("per-batch option count row=%d ids=%d want=%d", row, len(selection.PerBatch[row]), optionsPerRow)
		}
		if err := validateOptionIDs(selection.PerBatch[row], actions, fmt.Sprintf("row %d", row)); err != nil {
			return nil, err
		}
		ids[row] = append([]int(nil), selection.PerBatch[row]...)
	}
	return ids, nil
}

func validateOptionIDs(ids []int, actions int, label string) error {
	for i, id := range ids {
		if id < 0 || id >= actions {
			return fmt.Errorf("option selection %s id[%d]=%d out of range [0,%d)", label, i, id, actions)
		}
	}
	return nil
}

func (m *VisionActionScorer) optionRow(id int) []float32 {
	start := id * m.Config.Width
	return m.OptionEmbedding[start : start+m.Config.Width]
}

func (m *VisionActionScorer) positionRow(token int) []float32 {
	start := token * m.Config.Rank
	return m.Positions[start : start+m.Config.Rank]
}

func encodeSignedDifference(value float64) uint8 {
	rounded := math.RoundToEven(value) + 128.0
	if rounded < 0 {
		return 0
	}
	if rounded > 255 {
		return 255
	}
	return uint8(rounded)
}

func normalizedAttentionEntropy(attention []float32) float32 {
	if len(attention) <= 1 {
		return 0
	}
	var entropy float64
	for _, value := range attention {
		if value <= 0 {
			continue
		}
		entropy -= float64(value) * math.Log(float64(value))
	}
	return float32(entropy / math.Log(float64(len(attention))))
}

func make2DFloat32(rows, cols int) [][]float32 {
	out := make([][]float32, rows)
	for row := range out {
		out[row] = make([]float32, cols)
	}
	return out
}

func make3DFloat32(rows, cols, depth int) [][][]float32 {
	out := make([][][]float32, rows)
	for row := range out {
		out[row] = make([][]float32, cols)
		for col := range out[row] {
			out[row][col] = make([]float32, depth)
		}
	}
	return out
}

func clone2DFloat32(src [][]float32) [][]float32 {
	out := make([][]float32, len(src))
	for i := range src {
		out[i] = append([]float32(nil), src[i]...)
	}
	return out
}
