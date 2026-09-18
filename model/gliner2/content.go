package gliner2

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/internal/checked"
)

const spanContentMaskFloor = -float32(math.MaxFloat32 / 4)

// SpanContentPooler mirrors upstream boundary/content.py for single-document
// inference. Dropout is intentionally omitted because inference keeps it off.
type SpanContentPooler struct {
	HiddenSize      int       `json:"hidden_size"`
	ContentDim      int       `json:"content_dim"`
	UseSoftMaxPool  bool      `json:"use_soft_max_pool,omitempty"`
	ValueProjection Linear    `json:"value_projection"`
	LayerNorm       LayerNorm `json:"layer_norm"`
}

func (p SpanContentPooler) OutputDim() int {
	if p.UseSoftMaxPool {
		return 2 * p.ContentDim
	}
	return p.ContentDim
}

func (p SpanContentPooler) Validate() error {
	if p.HiddenSize <= 0 || p.ContentDim <= 0 {
		return fmt.Errorf("span content dims hidden=%d content=%d", p.HiddenSize, p.ContentDim)
	}
	if err := p.ValueProjection.Validate(); err != nil {
		return fmt.Errorf("span content value_projection: %w", err)
	}
	if p.ValueProjection.InDim != p.HiddenSize || p.ValueProjection.OutDim != p.ContentDim {
		return fmt.Errorf("span content value_projection dims out=%d in=%d want out=%d in=%d", p.ValueProjection.OutDim, p.ValueProjection.InDim, p.ContentDim, p.HiddenSize)
	}
	if err := p.LayerNorm.Validate(); err != nil {
		return fmt.Errorf("span content layer_norm: %w", err)
	}
	if p.LayerNorm.Dim() != p.OutputDim() {
		return fmt.Errorf("span content layer_norm dim=%d want=%d", p.LayerNorm.Dim(), p.OutputDim())
	}
	return nil
}

// LoadSpanContentPooler binds a content pooler from the exact module prefix
// provided by the caller, e.g. a scorer or shared candidate pool path.
func LoadSpanContentPooler(source TensorSource, hiddenSize, contentDim int, useSoftMaxPool bool, prefix string) (SpanContentPooler, error) {
	var p SpanContentPooler
	if source == nil || hiddenSize <= 0 || contentDim <= 0 {
		return p, fmt.Errorf("tensor source and positive hidden/content dims required")
	}
	if prefix == "" {
		return p, fmt.Errorf("span content pooler prefix required")
	}
	outputDim := contentDim
	if useSoftMaxPool {
		outputDim *= 2
	}
	r := weightReader{source: source}
	p = SpanContentPooler{
		HiddenSize:      hiddenSize,
		ContentDim:      contentDim,
		UseSoftMaxPool:  useSoftMaxPool,
		ValueProjection: r.linear(prefix+".value_projection", hiddenSize, contentDim),
		LayerNorm:       r.norm(prefix+".layer_norm", outputDim),
	}
	if r.err != nil {
		return SpanContentPooler{}, r.err
	}
	return p, p.Validate()
}

// BuildPrefix projects one padded document, masks invalid tokens, and returns
// the zero-prepended cumulative mean prefix together with the optional
// log-cumulative-exp prefix used for smooth max pooling.
func (p SpanContentPooler) BuildPrefix(textStates [][]float32, textMask []bool) ([][]float32, [][]float32, error) {
	if err := p.Validate(); err != nil {
		return nil, nil, err
	}
	if len(textStates) != len(textMask) {
		return nil, nil, fmt.Errorf("span content text_mask len=%d want=%d", len(textMask), len(textStates))
	}
	values, err := projectRows(p.ValueProjection, textStates, "span content text_states")
	if err != nil {
		return nil, nil, err
	}
	meanPrefix := makeMatrix(len(textStates)+1, p.ContentDim)
	var lsePrefix [][]float32
	if p.UseSoftMaxPool {
		lsePrefix = makeMatrix(len(textStates)+1, p.ContentDim)
		for i := range lsePrefix[0] {
			lsePrefix[0][i] = spanContentMaskFloor
		}
	}
	for row := range textStates {
		copy(meanPrefix[row+1], meanPrefix[row])
		if textMask[row] {
			for i := 0; i < p.ContentDim; i++ {
				meanPrefix[row+1][i] += values[row][i]
			}
		}
		if lsePrefix == nil {
			continue
		}
		for i := 0; i < p.ContentDim; i++ {
			value := spanContentMaskFloor
			if textMask[row] {
				value = values[row][i]
			}
			lsePrefix[row+1][i] = logAddExp32(lsePrefix[row][i], value)
		}
	}
	return meanPrefix, lsePrefix, nil
}

// Pool gathers half-open [start,end) span content for one document-pooled list
// of candidates, applies the optional smooth max branch, and layer-normalizes
// the final content representation.
func (p SpanContentPooler) PoolRows(meanPrefix, lsePrefix [][]float32, starts, ends []int) ([][]float32, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := p.validatePrefixes(meanPrefix, lsePrefix); err != nil {
		return nil, err
	}
	if len(starts) != len(ends) {
		return nil, fmt.Errorf("span content starts len=%d want=%d", len(starts), len(ends))
	}
	if len(starts) == 0 {
		return [][]float32{}, nil
	}
	outputDim := p.OutputDim()
	total, ok := checked.MulInt(len(starts), outputDim)
	if !ok {
		return nil, fmt.Errorf("span content row shape overflow rows=%d dim=%d", len(starts), outputDim)
	}
	raw := make([]float32, total)
	for i := range starts {
		p.poolRow(raw[i*outputDim:(i+1)*outputDim], meanPrefix, lsePrefix, starts[i], ends[i])
	}
	normed := make([]float32, total)
	if err := p.LayerNorm.ApplyBatch(raw, normed, len(starts)); err != nil {
		return nil, err
	}
	return rowsFromFlat(normed, len(starts), outputDim), nil
}

// Pool is the per-query variant for [query,candidate] half-open spans.
func (p SpanContentPooler) Pool(meanPrefix, lsePrefix [][]float32, starts, ends [][]int) ([][][]float32, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := p.validatePrefixes(meanPrefix, lsePrefix); err != nil {
		return nil, err
	}
	if len(starts) != len(ends) {
		return nil, fmt.Errorf("span content starts queries=%d want=%d", len(starts), len(ends))
	}
	out := make([][][]float32, len(starts))
	totalRows := 0
	for q := range starts {
		if len(starts[q]) != len(ends[q]) {
			return nil, fmt.Errorf("span content ends[%d] len=%d want=%d", q, len(ends[q]), len(starts[q]))
		}
		var ok bool
		totalRows, ok = checked.AddInt(totalRows, len(starts[q]))
		if !ok {
			return nil, fmt.Errorf("span content query rows overflow queries=%d", len(starts))
		}
		out[q] = make([][]float32, len(starts[q]))
	}
	if totalRows == 0 {
		return out, nil
	}
	outputDim := p.OutputDim()
	total, ok := checked.MulInt(totalRows, outputDim)
	if !ok {
		return nil, fmt.Errorf("span content state shape overflow rows=%d dim=%d", totalRows, outputDim)
	}
	raw := make([]float32, total)
	row := 0
	for q := range starts {
		for c := range starts[q] {
			p.poolRow(raw[row*outputDim:(row+1)*outputDim], meanPrefix, lsePrefix, starts[q][c], ends[q][c])
			row++
		}
	}
	normed := make([]float32, total)
	if err := p.LayerNorm.ApplyBatch(raw, normed, totalRows); err != nil {
		return nil, err
	}
	row = 0
	for q := range out {
		for c := range out[q] {
			state := make([]float32, outputDim)
			copy(state, normed[row*outputDim:(row+1)*outputDim])
			out[q][c] = state
			row++
		}
	}
	return out, nil
}

func (p SpanContentPooler) validatePrefixes(meanPrefix, lsePrefix [][]float32) error {
	if err := validateMatrix(meanPrefix, p.ContentDim, "span content mean_prefix"); err != nil {
		return err
	}
	if !p.UseSoftMaxPool {
		if len(lsePrefix) != 0 {
			return fmt.Errorf("span content lse_prefix provided but use_soft_max_pool is false")
		}
		return nil
	}
	if err := validateMatrix(lsePrefix, p.ContentDim, "span content lse_prefix"); err != nil {
		return err
	}
	if len(lsePrefix) != len(meanPrefix) {
		return fmt.Errorf("span content lse_prefix rows=%d want=%d", len(lsePrefix), len(meanPrefix))
	}
	return nil
}

func (p SpanContentPooler) poolRow(out []float32, meanPrefix, lsePrefix [][]float32, start, end int) {
	startClamped := clampIndex(start, len(meanPrefix))
	endClamped := clampIndex(end, len(meanPrefix))
	length := float32(max(end-start, 1))
	for i := 0; i < p.ContentDim; i++ {
		out[i] = (meanPrefix[endClamped][i] - meanPrefix[startClamped][i]) / length
	}
	if !p.UseSoftMaxPool {
		return
	}
	for i := 0; i < p.ContentDim; i++ {
		delta := lsePrefix[startClamped][i] - lsePrefix[endClamped][i]
		if delta > -1e-6 {
			delta = -1e-6
		}
		softMax := lsePrefix[endClamped][i] + float32(math.Log1p(-math.Exp(float64(delta))))
		out[p.ContentDim+i] = nanToZero32(softMax)
	}
}

func validateMatrix(rows [][]float32, cols int, name string) error {
	if len(rows) == 0 {
		return fmt.Errorf("%s empty", name)
	}
	for i, row := range rows {
		if len(row) != cols {
			return fmt.Errorf("%s row=%d len=%d want=%d", name, i, len(row), cols)
		}
	}
	return nil
}

func makeMatrix(rows, cols int) [][]float32 {
	out := make([][]float32, rows)
	for i := range out {
		out[i] = make([]float32, cols)
	}
	return out
}

func logAddExp32(a, b float32) float32 {
	if a < b {
		a, b = b, a
	}
	return a + float32(math.Log1p(math.Exp(float64(b-a))))
}

func nanToZero32(v float32) float32 {
	if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
		return 0
	}
	return v
}
