package gliner2

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
)

const (
	OverlapIdentical = iota
	OverlapNestedInside
	OverlapNestedOutside
	OverlapCrossing
	OverlapSameStart
	OverlapSameEnd
	OverlapDisjointLeft
	OverlapDisjointRight
	NumOverlapBuckets = 8
)

// ClassifyOverlapBucket maps one ordered half-open span pair to the upstream
// GLiNER overlap bucket. Precedence matches boundary/pool.py exactly so same
// boundary, same start, and same end cases stay distinct from containment.
func ClassifyOverlapBucket(left, right [2]int) int {
	s1, e1 := left[0], left[1]
	s2, e2 := right[0], right[1]
	bucket := OverlapCrossing
	if e1 <= s2 {
		bucket = OverlapDisjointLeft
	}
	if e2 <= s1 {
		bucket = OverlapDisjointRight
	}
	if s1 > s2 && e1 < e2 {
		bucket = OverlapNestedInside
	}
	if s1 < s2 && e1 > e2 {
		bucket = OverlapNestedOutside
	}
	if s1 == s2 && e1 != e2 {
		bucket = OverlapSameStart
	}
	if e1 == e2 && s1 != s2 {
		bucket = OverlapSameEnd
	}
	if s1 == s2 && e1 == e2 {
		bucket = OverlapIdentical
	}
	return bucket
}

// ClassifyOverlapBuckets computes the full ordered bucket matrix for one pooled
// candidate list.
func ClassifyOverlapBuckets(indices [][2]int) [][]int {
	out := make([][]int, len(indices))
	for i := range indices {
		out[i] = make([]int, len(indices))
		for j := range indices {
			out[i][j] = ClassifyOverlapBucket(indices[i], indices[j])
		}
	}
	return out
}

// OverlapBiasedCandidateAttention mirrors GLiNER2 candidate self-attention for
// one padded document pool. Dropout is omitted because inference keeps it off.
// The block keeps the upstream pre-norm attention + FFN structure.
type OverlapBiasedCandidateAttention struct {
	NumHeads            int       `json:"num_heads"`
	Norm1               LayerNorm `json:"norm1"`
	QKVProjection       Linear    `json:"qkv_projection"`
	OutputProjection    Linear    `json:"output_projection"`
	RelativeBias        []float32 `json:"relative_bias"`
	Norm2               LayerNorm `json:"norm2"`
	FFNInputProjection  Linear    `json:"ffn_input_projection"`
	FFNOutputProjection Linear    `json:"ffn_output_projection"`
}

func (a OverlapBiasedCandidateAttention) Validate() error {
	if a.NumHeads <= 0 {
		return fmt.Errorf("candidate attention num_heads=%d", a.NumHeads)
	}
	if err := a.Norm1.Validate(); err != nil {
		return fmt.Errorf("candidate attention norm1: %w", err)
	}
	if err := a.QKVProjection.Validate(); err != nil {
		return fmt.Errorf("candidate attention qkv_projection: %w", err)
	}
	if err := a.OutputProjection.Validate(); err != nil {
		return fmt.Errorf("candidate attention output_projection: %w", err)
	}
	if err := a.Norm2.Validate(); err != nil {
		return fmt.Errorf("candidate attention norm2: %w", err)
	}
	if err := a.FFNInputProjection.Validate(); err != nil {
		return fmt.Errorf("candidate attention ffn_input_projection: %w", err)
	}
	if err := a.FFNOutputProjection.Validate(); err != nil {
		return fmt.Errorf("candidate attention ffn_output_projection: %w", err)
	}
	dim := a.Norm1.Dim()
	if dim%a.NumHeads != 0 {
		return fmt.Errorf("candidate attention dim=%d not divisible by num_heads=%d", dim, a.NumHeads)
	}
	qkvDim, ok := checked.MulInt(3, dim)
	if !ok {
		return fmt.Errorf("candidate attention qkv dim overflow dim=%d", dim)
	}
	if a.QKVProjection.InDim != dim || a.QKVProjection.OutDim != qkvDim {
		return fmt.Errorf("candidate attention qkv_projection dims out=%d in=%d want out=%d in=%d", a.QKVProjection.OutDim, a.QKVProjection.InDim, qkvDim, dim)
	}
	if a.OutputProjection.InDim != dim || a.OutputProjection.OutDim != dim {
		return fmt.Errorf("candidate attention output_projection dims out=%d in=%d want out=%d in=%d", a.OutputProjection.OutDim, a.OutputProjection.InDim, dim, dim)
	}
	if a.Norm2.Dim() != dim {
		return fmt.Errorf("candidate attention norm2 dim=%d want=%d", a.Norm2.Dim(), dim)
	}
	biasLen, ok := checked.MulInt(NumOverlapBuckets, a.NumHeads)
	if !ok {
		return fmt.Errorf("candidate attention relative_bias overflow heads=%d", a.NumHeads)
	}
	if len(a.RelativeBias) != biasLen {
		return fmt.Errorf("candidate attention relative_bias len=%d want=%d", len(a.RelativeBias), biasLen)
	}
	ffnHidden, ok := checked.MulInt(4, dim)
	if !ok {
		return fmt.Errorf("candidate attention ffn dim overflow dim=%d", dim)
	}
	if a.FFNInputProjection.InDim != dim || a.FFNInputProjection.OutDim != ffnHidden {
		return fmt.Errorf("candidate attention ffn_input_projection dims out=%d in=%d want out=%d in=%d", a.FFNInputProjection.OutDim, a.FFNInputProjection.InDim, ffnHidden, dim)
	}
	if a.FFNOutputProjection.InDim != ffnHidden || a.FFNOutputProjection.OutDim != dim {
		return fmt.Errorf("candidate attention ffn_output_projection dims out=%d in=%d want out=%d in=%d", a.FFNOutputProjection.OutDim, a.FFNOutputProjection.InDim, dim, ffnHidden)
	}
	return nil
}

// LoadOverlapBiasedCandidateAttention binds one candidate attention block from
// the exact caller-provided module prefix, e.g.
// boundary_head.shared_pool_scorer.candidate_layers.0.
func LoadOverlapBiasedCandidateAttention(source TensorSource, dim, heads int, prefix string) (OverlapBiasedCandidateAttention, error) {
	var a OverlapBiasedCandidateAttention
	if source == nil || dim <= 0 || heads <= 0 {
		return a, fmt.Errorf("tensor source and positive dim/heads required")
	}
	if prefix == "" {
		return a, fmt.Errorf("candidate attention prefix required")
	}
	qkvDim, ok := checked.MulInt(3, dim)
	if !ok {
		return a, fmt.Errorf("candidate attention qkv dim overflow dim=%d", dim)
	}
	ffnHidden, ok := checked.MulInt(4, dim)
	if !ok {
		return a, fmt.Errorf("candidate attention ffn dim overflow dim=%d", dim)
	}
	r := weightReader{source: source}
	a = OverlapBiasedCandidateAttention{
		NumHeads:            heads,
		Norm1:               r.norm(prefix+".norm1", dim),
		QKVProjection:       r.linear(prefix+".qkv", dim, qkvDim),
		OutputProjection:    r.linear(prefix+".output", dim, dim),
		RelativeBias:        r.tensor(prefix+".relative_bias", NumOverlapBuckets, heads),
		Norm2:               r.norm(prefix+".norm2", dim),
		FFNInputProjection:  r.linear(prefix+".ffn.0", dim, ffnHidden),
		FFNOutputProjection: r.linear(prefix+".ffn.3", ffnHidden, dim),
	}
	if r.err != nil {
		return OverlapBiasedCandidateAttention{}, r.err
	}
	return a, a.Validate()
}

// Forward applies overlap-biased self-attention to one padded candidate list.
// Keys are visible when valid or when they are the query's diagonal position,
// which keeps every row softmax-safe even for fully masked padding rows.
func (a OverlapBiasedCandidateAttention) Forward(states [][]float32, indices [][2]int, mask []bool) ([][]float32, error) {
	if err := a.Validate(); err != nil {
		return nil, err
	}
	if len(states) != len(indices) {
		return nil, fmt.Errorf("candidate attention indices len=%d want=%d", len(indices), len(states))
	}
	if len(states) != len(mask) {
		return nil, fmt.Errorf("candidate attention mask len=%d want=%d", len(mask), len(states))
	}
	rows := len(states)
	dim := a.Norm1.Dim()
	flat, err := flattenRows(states, dim, "candidate attention states")
	if err != nil {
		return nil, err
	}
	if rows == 0 {
		return [][]float32{}, nil
	}
	normed := make([]float32, len(flat))
	if err := a.Norm1.ApplyBatch(flat, normed, rows); err != nil {
		return nil, err
	}
	qkvDim := a.QKVProjection.OutDim
	qkv := make([]float32, rows*qkvDim)
	if err := a.QKVProjection.ApplyBatch(normed, qkv, rows); err != nil {
		return nil, err
	}
	buckets := ClassifyOverlapBuckets(indices)
	headDim := dim / a.NumHeads
	scale := float32(1 / math.Sqrt(float64(headDim)))
	negInf := float32(math.Inf(-1))
	context := make([]float32, len(flat))
	scores := make([]float32, rows)
	for query := 0; query < rows; query++ {
		qRowBase := query * qkvDim
		ctxRowBase := query * dim
		for head := 0; head < a.NumHeads; head++ {
			qHeadBase := qRowBase + head*headDim
			qHead := qkv[qHeadBase : qHeadBase+headDim]
			for key := 0; key < rows; key++ {
				allowed := mask[key] || key == query
				if !allowed {
					scores[key] = negInf
					continue
				}
				kHeadBase := key*qkvDim + dim + head*headDim
				scores[key] = simd.Sdot(qHead, qkv[kHeadBase:kHeadBase+headDim])*scale + a.RelativeBias[buckets[query][key]*a.NumHeads+head]
			}
			if !simd.SoftmaxInPlace(scores) {
				return nil, fmt.Errorf("candidate attention softmax rejected rows=%d", rows)
			}
			outHead := context[ctxRowBase+head*headDim : ctxRowBase+(head+1)*headDim]
			for i := range outHead {
				outHead[i] = 0
			}
			for key := 0; key < rows; key++ {
				weight := scores[key]
				if weight == 0 {
					continue
				}
				vHeadBase := key*qkvDim + 2*dim + head*headDim
				vHead := qkv[vHeadBase : vHeadBase+headDim]
				for i := 0; i < headDim; i++ {
					outHead[i] += weight * vHead[i]
				}
			}
		}
	}
	attnUpdate := make([]float32, len(flat))
	if err := a.OutputProjection.ApplyBatch(context, attnUpdate, rows); err != nil {
		return nil, err
	}
	attended := make([]float32, len(flat))
	copy(attended, flat)
	for i := range attended {
		attended[i] += attnUpdate[i]
	}
	normed2 := make([]float32, len(flat))
	if err := a.Norm2.ApplyBatch(attended, normed2, rows); err != nil {
		return nil, err
	}
	ffnHidden := make([]float32, rows*a.FFNInputProjection.OutDim)
	if err := a.FFNInputProjection.ApplyBatch(normed2, ffnHidden, rows); err != nil {
		return nil, err
	}
	for i := range ffnHidden {
		ffnHidden[i] = gelu32(ffnHidden[i])
	}
	ffnUpdate := make([]float32, len(flat))
	if err := a.FFNOutputProjection.ApplyBatch(ffnHidden, ffnUpdate, rows); err != nil {
		return nil, err
	}
	out := make([]float32, len(flat))
	copy(out, attended)
	for i := range out {
		out[i] += ffnUpdate[i]
	}
	for row := 0; row < rows; row++ {
		if mask[row] {
			continue
		}
		clear(out[row*dim : (row+1)*dim])
	}
	return rowsFromFlat(out, rows, dim), nil
}

func gelu32(x float32) float32 {
	return 0.5 * x * (1 + float32(math.Erf(float64(x)/math.Sqrt2)))
}
