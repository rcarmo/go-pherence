package gliner2

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
)

const defaultLayerNormEpsilon = float32(1e-5)

// BoundaryEncoding holds the refined per-boundary states for one padded
// document together with the validity mask for boundaries 0..len.
type BoundaryEncoding struct {
	States [][]float32 `json:"states"`
	Mask   []bool      `json:"mask"`
}

// Linear is a checked dense row-major projection matching torch.nn.Linear:
// out = x @ weight^T + bias.
type Linear struct {
	InDim  int       `json:"in_dim"`
	OutDim int       `json:"out_dim"`
	Weight []float32 `json:"weight"`
	Bias   []float32 `json:"bias,omitempty"`
}

func (l Linear) Validate() error {
	if l.InDim <= 0 || l.OutDim <= 0 {
		return fmt.Errorf("linear invalid dims out=%d in=%d", l.OutDim, l.InDim)
	}
	weightLen, ok := checked.MulInt(l.OutDim, l.InDim)
	if !ok {
		return fmt.Errorf("linear dims overflow out=%d in=%d", l.OutDim, l.InDim)
	}
	if len(l.Weight) != weightLen {
		return fmt.Errorf("linear weight len=%d want=%d", len(l.Weight), weightLen)
	}
	if len(l.Bias) != 0 && len(l.Bias) != l.OutDim {
		return fmt.Errorf("linear bias len=%d want=%d", len(l.Bias), l.OutDim)
	}
	return nil
}

// Apply computes out[OutDim] = W[OutDim,InDim] · x[InDim] + bias.
func (l Linear) Apply(x, out []float32) error {
	if err := l.Validate(); err != nil {
		return err
	}
	if len(x) < l.InDim || len(out) < l.OutDim {
		return fmt.Errorf("linear buffers x=%d/%d out=%d/%d", len(x), l.InDim, len(out), l.OutDim)
	}
	if !simd.GemvRows(out[:l.OutDim], x[:l.InDim], l.Weight, l.OutDim, l.InDim) {
		return fmt.Errorf("linear GEMV rejected out=%d in=%d", l.OutDim, l.InDim)
	}
	if len(l.Bias) != 0 {
		for i := 0; i < l.OutDim; i++ {
			out[i] += l.Bias[i]
		}
	}
	return nil
}

// ApplyBatch computes row-major out[batch,OutDim] = x[batch,InDim] @ W^T + bias.
func (l Linear) ApplyBatch(x, out []float32, batch int) error {
	if err := l.Validate(); err != nil {
		return err
	}
	if batch < 0 {
		return fmt.Errorf("linear invalid batch=%d", batch)
	}
	if batch == 0 {
		return nil
	}
	xLen, okX := checked.MulInt(batch, l.InDim)
	outLen, okOut := checked.MulInt(batch, l.OutDim)
	if !okX || !okOut {
		return fmt.Errorf("linear batch overflow batch=%d out=%d in=%d", batch, l.OutDim, l.InDim)
	}
	if len(x) < xLen || len(out) < outLen {
		return fmt.Errorf("linear batch buffers x=%d/%d out=%d/%d", len(x), xLen, len(out), outLen)
	}
	if batch == 1 {
		return l.Apply(x[:l.InDim], out[:l.OutDim])
	}
	if !simd.GemmRows(out[:outLen], x[:xLen], l.Weight, batch, l.OutDim, l.InDim) {
		return fmt.Errorf("linear GEMM rejected batch=%d out=%d in=%d", batch, l.OutDim, l.InDim)
	}
	if len(l.Bias) != 0 {
		for b := 0; b < batch; b++ {
			row := out[b*l.OutDim : (b+1)*l.OutDim]
			for i := 0; i < l.OutDim; i++ {
				row[i] += l.Bias[i]
			}
		}
	}
	return nil
}

// LayerNorm is an affine last-axis layer normalization.
type LayerNorm struct {
	Weight  []float32 `json:"weight"`
	Bias    []float32 `json:"bias"`
	Epsilon float32   `json:"epsilon,omitempty"`
}

func (l LayerNorm) Dim() int { return len(l.Weight) }

func (l LayerNorm) epsilon() float32 {
	if l.Epsilon == 0 {
		return defaultLayerNormEpsilon
	}
	return l.Epsilon
}

func (l LayerNorm) Validate() error {
	if len(l.Weight) == 0 {
		return fmt.Errorf("layernorm empty weight")
	}
	if len(l.Bias) != len(l.Weight) {
		return fmt.Errorf("layernorm bias len=%d want=%d", len(l.Bias), len(l.Weight))
	}
	eps := l.epsilon()
	if eps <= 0 || math.IsNaN(float64(eps)) || math.IsInf(float64(eps), 0) {
		return fmt.Errorf("layernorm epsilon=%g", eps)
	}
	return nil
}

// ApplyBatch applies layer normalization to row-major x[rows,Dim] into out.
func (l LayerNorm) ApplyBatch(x, out []float32, rows int) error {
	if err := l.Validate(); err != nil {
		return err
	}
	if rows < 0 {
		return fmt.Errorf("layernorm invalid rows=%d", rows)
	}
	if rows == 0 {
		return nil
	}
	total, ok := checked.MulInt(rows, len(l.Weight))
	if !ok {
		return fmt.Errorf("layernorm rows overflow rows=%d dim=%d", rows, len(l.Weight))
	}
	if len(x) < total || len(out) < total {
		return fmt.Errorf("layernorm buffers x=%d/%d out=%d/%d", len(x), total, len(out), total)
	}
	if !simd.LayerNormLastAxisTo(out[:total], x[:total], rows, len(l.Weight), l.Weight, l.Bias, l.epsilon()) {
		return fmt.Errorf("layernorm rejected rows=%d dim=%d", rows, len(l.Weight))
	}
	return nil
}

// ResidualSwiGLU is the GLiNER2.5 compact pre-norm feed-forward block.
type ResidualSwiGLU struct {
	Norm             LayerNorm `json:"norm"`
	InputProjection  Linear    `json:"input_projection"`
	OutputProjection Linear    `json:"output_projection"`
}

func (r ResidualSwiGLU) Validate() error {
	if err := r.Norm.Validate(); err != nil {
		return fmt.Errorf("swiglu norm: %w", err)
	}
	if err := r.InputProjection.Validate(); err != nil {
		return fmt.Errorf("swiglu input_projection: %w", err)
	}
	if err := r.OutputProjection.Validate(); err != nil {
		return fmt.Errorf("swiglu output_projection: %w", err)
	}
	dim := r.Norm.Dim()
	if r.InputProjection.InDim != dim {
		return fmt.Errorf("swiglu input_projection in=%d want=%d", r.InputProjection.InDim, dim)
	}
	if r.InputProjection.OutDim <= 0 || r.InputProjection.OutDim%2 != 0 {
		return fmt.Errorf("swiglu input_projection out=%d must be positive even", r.InputProjection.OutDim)
	}
	hidden := r.InputProjection.OutDim / 2
	if r.OutputProjection.InDim != hidden || r.OutputProjection.OutDim != dim {
		return fmt.Errorf("swiglu output_projection dims out=%d in=%d want out=%d in=%d", r.OutputProjection.OutDim, r.OutputProjection.InDim, dim, hidden)
	}
	return nil
}

// Forward applies the residual SwiGLU block to one [rows,dim] state matrix.
func (r ResidualSwiGLU) Forward(states [][]float32) ([][]float32, error) {
	if err := r.Validate(); err != nil {
		return nil, err
	}
	rows := len(states)
	dim := r.Norm.Dim()
	flat, err := flattenRows(states, dim, "swiglu states")
	if err != nil {
		return nil, err
	}
	if rows == 0 {
		return [][]float32{}, nil
	}
	normed := make([]float32, len(flat))
	if err := r.Norm.ApplyBatch(flat, normed, rows); err != nil {
		return nil, err
	}
	hidden := r.InputProjection.OutDim / 2
	projected := make([]float32, rows*r.InputProjection.OutDim)
	if err := r.InputProjection.ApplyBatch(normed, projected, rows); err != nil {
		return nil, err
	}
	activated := make([]float32, rows*hidden)
	for row := 0; row < rows; row++ {
		base := row * r.InputProjection.OutDim
		value := projected[base : base+hidden]
		gate := projected[base+hidden : base+2*hidden]
		out := activated[row*hidden : (row+1)*hidden]
		for i := 0; i < hidden; i++ {
			out[i] = value[i] * silu(gate[i])
		}
	}
	update := make([]float32, rows*dim)
	if err := r.OutputProjection.ApplyBatch(activated, update, rows); err != nil {
		return nil, err
	}
	out := make([]float32, len(flat))
	copy(out, flat)
	for i := range out {
		out[i] += update[i]
	}
	return rowsFromFlat(out, rows, dim), nil
}

// BoundaryAttentionBlock is GLiNER2.5 pre-norm self-attention over valid
// boundary positions, with optional symmetric local windowing.
type BoundaryAttentionBlock struct {
	NumHeads         int       `json:"num_heads"`
	Window           int       `json:"window"`
	Norm             LayerNorm `json:"norm"`
	QKVProjection    Linear    `json:"qkv_projection"`
	OutputProjection Linear    `json:"output_projection"`
}

func (b BoundaryAttentionBlock) Validate() error {
	if err := b.Norm.Validate(); err != nil {
		return fmt.Errorf("attention norm: %w", err)
	}
	if err := b.QKVProjection.Validate(); err != nil {
		return fmt.Errorf("attention qkv_projection: %w", err)
	}
	if err := b.OutputProjection.Validate(); err != nil {
		return fmt.Errorf("attention output_projection: %w", err)
	}
	if b.NumHeads <= 0 {
		return fmt.Errorf("attention num_heads=%d", b.NumHeads)
	}
	if b.Window < 0 {
		return fmt.Errorf("attention window=%d", b.Window)
	}
	dim := b.Norm.Dim()
	if dim%b.NumHeads != 0 {
		return fmt.Errorf("attention dim=%d not divisible by num_heads=%d", dim, b.NumHeads)
	}
	qkvOut, ok := checked.MulInt(3, dim)
	if !ok {
		return fmt.Errorf("attention qkv dim overflow dim=%d", dim)
	}
	if b.QKVProjection.InDim != dim || b.QKVProjection.OutDim != qkvOut {
		return fmt.Errorf("attention qkv_projection dims out=%d in=%d want out=%d in=%d", b.QKVProjection.OutDim, b.QKVProjection.InDim, qkvOut, dim)
	}
	if b.OutputProjection.InDim != dim || b.OutputProjection.OutDim != dim {
		return fmt.Errorf("attention output_projection dims out=%d in=%d want out=%d in=%d", b.OutputProjection.OutDim, b.OutputProjection.InDim, dim, dim)
	}
	return nil
}

// Forward applies non-causal self-attention to one [boundaries,dim] document.
func (b BoundaryAttentionBlock) Forward(states [][]float32, mask []bool) ([][]float32, error) {
	if err := b.Validate(); err != nil {
		return nil, err
	}
	if len(states) != len(mask) {
		return nil, fmt.Errorf("attention mask len=%d want=%d", len(mask), len(states))
	}
	rows := len(states)
	dim := b.Norm.Dim()
	flat, err := flattenRows(states, dim, "attention states")
	if err != nil {
		return nil, err
	}
	if rows == 0 {
		return [][]float32{}, nil
	}
	normed := make([]float32, len(flat))
	if err := b.Norm.ApplyBatch(flat, normed, rows); err != nil {
		return nil, err
	}
	qkv := make([]float32, rows*b.QKVProjection.OutDim)
	if err := b.QKVProjection.ApplyBatch(normed, qkv, rows); err != nil {
		return nil, err
	}
	q := make([]float32, rows*dim)
	k := make([]float32, rows*dim)
	v := make([]float32, rows*dim)
	for row := 0; row < rows; row++ {
		base := row * 3 * dim
		copy(q[row*dim:(row+1)*dim], qkv[base:base+dim])
		copy(k[row*dim:(row+1)*dim], qkv[base+dim:base+2*dim])
		copy(v[row*dim:(row+1)*dim], qkv[base+2*dim:base+3*dim])
	}
	headDim := dim / b.NumHeads
	scale := float32(1 / math.Sqrt(float64(headDim)))
	context := make([]float32, rows*dim)
	scores := make([]float32, rows)
	for query := 0; query < rows; query++ {
		for head := 0; head < b.NumHeads; head++ {
			qBase := query*dim + head*headDim
			qHead := q[qBase : qBase+headDim]
			for key := 0; key < rows; key++ {
				allowed := mask[key] && (b.Window == 0 || absInt(query-key) <= b.Window)
				if query == key {
					allowed = true
				}
				if !allowed {
					scores[key] = float32(math.Inf(-1))
					continue
				}
				kBase := key*dim + head*headDim
				scores[key] = simd.Sdot(qHead, k[kBase:kBase+headDim]) * scale
			}
			if !simd.SoftmaxInPlace(scores) {
				return nil, fmt.Errorf("attention softmax rejected rows=%d", rows)
			}
			outHead := context[qBase : qBase+headDim]
			for i := range outHead {
				outHead[i] = 0
			}
			for key := 0; key < rows; key++ {
				weight := scores[key]
				if weight == 0 {
					continue
				}
				vBase := key*dim + head*headDim
				vHead := v[vBase : vBase+headDim]
				for i := 0; i < headDim; i++ {
					outHead[i] += weight * vHead[i]
				}
			}
		}
	}
	update := make([]float32, rows*dim)
	if err := b.OutputProjection.ApplyBatch(context, update, rows); err != nil {
		return nil, err
	}
	out := make([]float32, len(flat))
	copy(out, flat)
	for row := 0; row < rows; row++ {
		rowOut := out[row*dim : (row+1)*dim]
		rowUpdate := update[row*dim : (row+1)*dim]
		if mask[row] {
			for i := 0; i < dim; i++ {
				rowOut[i] += rowUpdate[i]
			}
			continue
		}
		for i := 0; i < dim; i++ {
			rowOut[i] = 0
		}
	}
	return rowsFromFlat(out, rows, dim), nil
}

// BoundaryEncoder projects and refines token states into per-boundary states.
type BoundaryEncoder struct {
	HiddenSize       int                      `json:"hidden_size"`
	BoundaryDim      int                      `json:"boundary_dim"`
	LeftProjection   Linear                   `json:"left_projection"`
	RightProjection  Linear                   `json:"right_projection"`
	OutputProjection Linear                   `json:"output_projection"`
	LayerNorm        LayerNorm                `json:"layer_norm"`
	AttentionBlocks  []BoundaryAttentionBlock `json:"attention_blocks,omitempty"`
	RefinementBlocks []ResidualSwiGLU         `json:"refinement_blocks,omitempty"`
	BosState         []float32                `json:"bos_state"`
	EosState         []float32                `json:"eos_state"`
}

func (e BoundaryEncoder) Validate() error {
	if e.HiddenSize <= 0 || e.BoundaryDim <= 0 {
		return fmt.Errorf("boundary encoder dims hidden=%d boundary=%d", e.HiddenSize, e.BoundaryDim)
	}
	if len(e.BosState) != e.HiddenSize {
		return fmt.Errorf("boundary encoder bos_state len=%d want=%d", len(e.BosState), e.HiddenSize)
	}
	if len(e.EosState) != e.HiddenSize {
		return fmt.Errorf("boundary encoder eos_state len=%d want=%d", len(e.EosState), e.HiddenSize)
	}
	if err := e.LeftProjection.Validate(); err != nil {
		return fmt.Errorf("boundary encoder left_projection: %w", err)
	}
	if err := e.RightProjection.Validate(); err != nil {
		return fmt.Errorf("boundary encoder right_projection: %w", err)
	}
	if err := e.OutputProjection.Validate(); err != nil {
		return fmt.Errorf("boundary encoder output_projection: %w", err)
	}
	if err := e.LayerNorm.Validate(); err != nil {
		return fmt.Errorf("boundary encoder layer_norm: %w", err)
	}
	if e.LeftProjection.InDim != e.HiddenSize || e.LeftProjection.OutDim != e.BoundaryDim {
		return fmt.Errorf("boundary encoder left_projection dims out=%d in=%d want out=%d in=%d", e.LeftProjection.OutDim, e.LeftProjection.InDim, e.BoundaryDim, e.HiddenSize)
	}
	if e.RightProjection.InDim != e.HiddenSize || e.RightProjection.OutDim != e.BoundaryDim {
		return fmt.Errorf("boundary encoder right_projection dims out=%d in=%d want out=%d in=%d", e.RightProjection.OutDim, e.RightProjection.InDim, e.BoundaryDim, e.HiddenSize)
	}
	outputIn, ok := checked.MulInt(2, e.BoundaryDim)
	if !ok {
		return fmt.Errorf("boundary encoder output_projection dim overflow boundary=%d", e.BoundaryDim)
	}
	if e.OutputProjection.InDim != outputIn || e.OutputProjection.OutDim != e.BoundaryDim {
		return fmt.Errorf("boundary encoder output_projection dims out=%d in=%d want out=%d in=%d", e.OutputProjection.OutDim, e.OutputProjection.InDim, e.BoundaryDim, outputIn)
	}
	if e.LayerNorm.Dim() != e.BoundaryDim {
		return fmt.Errorf("boundary encoder layer_norm dim=%d want=%d", e.LayerNorm.Dim(), e.BoundaryDim)
	}
	for i, block := range e.AttentionBlocks {
		if err := block.Validate(); err != nil {
			return fmt.Errorf("boundary encoder attention_blocks[%d]: %w", i, err)
		}
		if block.Norm.Dim() != e.BoundaryDim {
			return fmt.Errorf("boundary encoder attention_blocks[%d] dim=%d want=%d", i, block.Norm.Dim(), e.BoundaryDim)
		}
	}
	for i, block := range e.RefinementBlocks {
		if err := block.Validate(); err != nil {
			return fmt.Errorf("boundary encoder refinement_blocks[%d]: %w", i, err)
		}
		if block.Norm.Dim() != e.BoundaryDim {
			return fmt.Errorf("boundary encoder refinement_blocks[%d] dim=%d want=%d", i, block.Norm.Dim(), e.BoundaryDim)
		}
	}
	return nil
}

// BuildBoundaryMask returns the validity mask for boundaries 0..maxTextLength,
// where boundary i is valid iff i <= validLength.
func BuildBoundaryMask(validLength, maxTextLength int) ([]bool, error) {
	if maxTextLength < 0 {
		return nil, fmt.Errorf("boundary mask negative max_text_length=%d", maxTextLength)
	}
	if validLength < 0 || validLength > maxTextLength {
		return nil, fmt.Errorf("boundary mask valid_length=%d outside [0,%d]", validLength, maxTextLength)
	}
	mask := make([]bool, maxTextLength+1)
	for i := 0; i <= validLength; i++ {
		mask[i] = true
	}
	return mask, nil
}

// Forward encodes one padded document. textStates is a padded [L,H] matrix and
// validLength is the number of real tokens (0 <= validLength <= L).
func (e BoundaryEncoder) Forward(textStates [][]float32, validLength int) (BoundaryEncoding, error) {
	if err := e.Validate(); err != nil {
		return BoundaryEncoding{}, err
	}
	if validLength < 0 || validLength > len(textStates) {
		return BoundaryEncoding{}, fmt.Errorf("boundary encoder valid_length=%d outside [0,%d]", validLength, len(textStates))
	}
	textFlat, err := flattenRows(textStates, e.HiddenSize, "boundary encoder text_states")
	if err != nil {
		return BoundaryEncoding{}, err
	}
	textLen := len(textStates)
	mask, err := BuildBoundaryMask(validLength, textLen)
	if err != nil {
		return BoundaryEncoding{}, err
	}
	left, err := shiftLeftWithBOS(textFlat, textLen, e.HiddenSize, e.BosState)
	if err != nil {
		return BoundaryEncoding{}, err
	}
	right, err := shiftRightWithEOS(textFlat, textLen, e.HiddenSize, validLength, e.EosState)
	if err != nil {
		return BoundaryEncoding{}, err
	}
	rows := textLen + 1
	leftProjected := make([]float32, rows*e.BoundaryDim)
	rightProjected := make([]float32, rows*e.BoundaryDim)
	if err := e.LeftProjection.ApplyBatch(left, leftProjected, rows); err != nil {
		return BoundaryEncoding{}, err
	}
	if err := e.RightProjection.ApplyBatch(right, rightProjected, rows); err != nil {
		return BoundaryEncoding{}, err
	}
	concatDim := 2 * e.BoundaryDim
	concat := make([]float32, rows*concatDim)
	for row := 0; row < rows; row++ {
		baseOut := row * concatDim
		copy(concat[baseOut:baseOut+e.BoundaryDim], leftProjected[row*e.BoundaryDim:(row+1)*e.BoundaryDim])
		copy(concat[baseOut+e.BoundaryDim:baseOut+concatDim], rightProjected[row*e.BoundaryDim:(row+1)*e.BoundaryDim])
	}
	projected := make([]float32, rows*e.BoundaryDim)
	if err := e.OutputProjection.ApplyBatch(concat, projected, rows); err != nil {
		return BoundaryEncoding{}, err
	}
	normed := make([]float32, len(projected))
	if err := e.LayerNorm.ApplyBatch(projected, normed, rows); err != nil {
		return BoundaryEncoding{}, err
	}
	states := rowsFromFlat(normed, rows, e.BoundaryDim)
	for _, block := range e.AttentionBlocks {
		states, err = block.Forward(states, mask)
		if err != nil {
			return BoundaryEncoding{}, err
		}
	}
	for _, block := range e.RefinementBlocks {
		states, err = block.Forward(states)
		if err != nil {
			return BoundaryEncoding{}, err
		}
	}
	for row := validLength + 1; row < len(states); row++ {
		for i := range states[row] {
			states[row][i] = 0
		}
	}
	return BoundaryEncoding{States: states, Mask: append([]bool(nil), mask...)}, nil
}

func shiftLeftWithBOS(textFlat []float32, rows, hidden int, bosState []float32) ([]float32, error) {
	need, ok := checked.MulInt(rows, hidden)
	if !ok {
		return nil, fmt.Errorf("shift left rows overflow rows=%d hidden=%d", rows, hidden)
	}
	if len(textFlat) != need {
		return nil, fmt.Errorf("shift left text len=%d want=%d", len(textFlat), need)
	}
	if len(bosState) != hidden {
		return nil, fmt.Errorf("shift left bos len=%d want=%d", len(bosState), hidden)
	}
	out := make([]float32, (rows+1)*hidden)
	copy(out[:hidden], bosState)
	copy(out[hidden:], textFlat)
	return out, nil
}

func shiftRightWithEOS(textFlat []float32, rows, hidden, validLength int, eosState []float32) ([]float32, error) {
	need, ok := checked.MulInt(rows, hidden)
	if !ok {
		return nil, fmt.Errorf("shift right rows overflow rows=%d hidden=%d", rows, hidden)
	}
	if len(textFlat) != need {
		return nil, fmt.Errorf("shift right text len=%d want=%d", len(textFlat), need)
	}
	if validLength < 0 || validLength > rows {
		return nil, fmt.Errorf("shift right valid_length=%d outside [0,%d]", validLength, rows)
	}
	if len(eosState) != hidden {
		return nil, fmt.Errorf("shift right eos len=%d want=%d", len(eosState), hidden)
	}
	out := make([]float32, (rows+1)*hidden)
	copy(out[:rows*hidden], textFlat)
	copy(out[rows*hidden:(rows+1)*hidden], eosState)
	copy(out[validLength*hidden:(validLength+1)*hidden], eosState)
	return out, nil
}

func flattenRows(rows [][]float32, cols int, name string) ([]float32, error) {
	if cols <= 0 {
		return nil, fmt.Errorf("%s invalid cols=%d", name, cols)
	}
	total, ok := checked.MulInt(len(rows), cols)
	if !ok {
		return nil, fmt.Errorf("%s shape overflow rows=%d cols=%d", name, len(rows), cols)
	}
	flat := make([]float32, total)
	for i, row := range rows {
		if len(row) != cols {
			return nil, fmt.Errorf("%s row=%d len=%d want=%d", name, i, len(row), cols)
		}
		copy(flat[i*cols:(i+1)*cols], row)
	}
	return flat, nil
}

func rowsFromFlat(flat []float32, rows, cols int) [][]float32 {
	out := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		row := make([]float32, cols)
		copy(row, flat[i*cols:(i+1)*cols])
		out[i] = row
	}
	return out
}

func silu(x float32) float32 {
	if x >= 0 {
		return x / (1 + float32(math.Exp(-float64(x))))
	}
	e := float32(math.Exp(float64(x)))
	return x * e / (1 + e)
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
