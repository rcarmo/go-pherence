package gliner2

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// RecordHead mirrors the native primitives of upstream boundary/records.py.
// It only binds the shared projections used to form anchorless instance states,
// score object/latent seeds, and compute null-aware field assignment logits.
type RecordHead struct {
	HiddenSize      int       `json:"hidden_size"`
	RecordDim       int       `json:"record_dim"`
	InstanceQueries int       `json:"instance_queries"`
	InstProjection  Linear    `json:"inst_proj"`
	FieldProjection Linear    `json:"field_proj"`
	CandProjection  Linear    `json:"cand_proj"`
	NullEmbed       []float32 `json:"null_embed"`
	ObjectHead      Linear    `json:"object_head"`
	LatentSeedHead  Linear    `json:"latent_seed_head"`
	InstanceEmbed   []float32 `json:"instance_embed"`
	QProjection     Linear    `json:"q_proj"`
	KProjection     Linear    `json:"k_proj"`
	VProjection     Linear    `json:"v_proj"`
}

func (h RecordHead) Validate() error {
	if h.HiddenSize <= 0 || h.RecordDim <= 0 || h.InstanceQueries <= 0 {
		return fmt.Errorf("record head dims hidden=%d record=%d instances=%d", h.HiddenSize, h.RecordDim, h.InstanceQueries)
	}
	for name, projection := range map[string]Linear{
		"inst_proj":        h.InstProjection,
		"field_proj":       h.FieldProjection,
		"cand_proj":        h.CandProjection,
		"object_head":      h.ObjectHead,
		"latent_seed_head": h.LatentSeedHead,
		"q_proj":           h.QProjection,
		"k_proj":           h.KProjection,
		"v_proj":           h.VProjection,
	} {
		if err := projection.Validate(); err != nil {
			return fmt.Errorf("record head %s: %w", name, err)
		}
	}
	for name, projection := range map[string]Linear{
		"inst_proj":  h.InstProjection,
		"field_proj": h.FieldProjection,
		"cand_proj":  h.CandProjection,
		"q_proj":     h.QProjection,
		"k_proj":     h.KProjection,
	} {
		if projection.InDim != h.HiddenSize || projection.OutDim != h.RecordDim {
			return fmt.Errorf("record head %s dims out=%d in=%d want out=%d in=%d", name, projection.OutDim, projection.InDim, h.RecordDim, h.HiddenSize)
		}
	}
	if h.ObjectHead.InDim != h.HiddenSize || h.ObjectHead.OutDim != 1 {
		return fmt.Errorf("record head object_head dims out=%d in=%d want out=%d in=%d", h.ObjectHead.OutDim, h.ObjectHead.InDim, 1, h.HiddenSize)
	}
	if h.LatentSeedHead.InDim != h.HiddenSize || h.LatentSeedHead.OutDim != 1 {
		return fmt.Errorf("record head latent_seed_head dims out=%d in=%d want out=%d in=%d", h.LatentSeedHead.OutDim, h.LatentSeedHead.InDim, 1, h.HiddenSize)
	}
	if h.VProjection.InDim != h.HiddenSize || h.VProjection.OutDim != h.HiddenSize {
		return fmt.Errorf("record head v_proj dims out=%d in=%d want out=%d in=%d", h.VProjection.OutDim, h.VProjection.InDim, h.HiddenSize, h.HiddenSize)
	}
	if len(h.NullEmbed) != h.RecordDim {
		return fmt.Errorf("record head null_embed len=%d want=%d", len(h.NullEmbed), h.RecordDim)
	}
	wantEmbed := h.InstanceQueries * h.HiddenSize
	if len(h.InstanceEmbed) != wantEmbed {
		return fmt.Errorf("record head instance_embed len=%d want=%d", len(h.InstanceEmbed), wantEmbed)
	}
	return nil
}

// LoadRecordHead binds the published record decoder primitives from the exact
// record_decoder.* safetensors prefixes used by GLiNER 2.5 boundary models.
func LoadRecordHead(source TensorSource, hiddenSize int, c BoundaryHeadConfig) (RecordHead, error) {
	var h RecordHead
	if source == nil || hiddenSize <= 0 {
		return h, fmt.Errorf("tensor source and hidden width required")
	}
	if err := c.Validate(); err != nil {
		return h, err
	}
	r := weightReader{source: source}
	prefix := "record_decoder"
	h = RecordHead{
		HiddenSize:      hiddenSize,
		RecordDim:       c.RecordDim,
		InstanceQueries: c.RecordInstanceQueries,
		InstProjection:  r.linear(prefix+".inst_proj", hiddenSize, c.RecordDim),
		FieldProjection: r.linear(prefix+".field_proj", hiddenSize, c.RecordDim),
		CandProjection:  r.linear(prefix+".cand_proj", hiddenSize, c.RecordDim),
		NullEmbed:       r.tensor(prefix+".null_embed", c.RecordDim),
		ObjectHead:      r.linear(prefix+".object_head", hiddenSize, 1),
		LatentSeedHead:  r.linear(prefix+".latent_seed_head", hiddenSize, 1),
		InstanceEmbed:   r.tensor(prefix+".instance_embed", c.RecordInstanceQueries, hiddenSize),
		QProjection:     r.linear(prefix+".q_proj", hiddenSize, c.RecordDim),
		KProjection:     r.linear(prefix+".k_proj", hiddenSize, c.RecordDim),
		VProjection:     r.linear(prefix+".v_proj", hiddenSize, hiddenSize),
	}
	if r.err != nil {
		return RecordHead{}, r.err
	}
	return h, h.Validate()
}

// ObjectLogits applies the object existence head to instance states [Ni,H].
func (h RecordHead) ObjectLogits(instStates [][]float32) ([]float32, error) {
	if err := h.Validate(); err != nil {
		return nil, err
	}
	return h.applyScalarHead(h.ObjectHead, instStates, "record head instance states")
}

// LatentSeedLogits applies the latent seed head to candidate states [C,H].
func (h RecordHead) LatentSeedLogits(candidateStates [][]float32) ([]float32, error) {
	if err := h.Validate(); err != nil {
		return nil, err
	}
	return h.applyScalarHead(h.LatentSeedHead, candidateStates, "record head candidate states")
}

func (h RecordHead) applyScalarHead(head Linear, states [][]float32, name string) ([]float32, error) {
	flat, err := flattenRows(states, head.InDim, name)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(states))
	if err := head.ApplyBatch(flat, out, len(states)); err != nil {
		return nil, err
	}
	return out, nil
}

// AssignLogits mirrors RecordHead._assign_logits. Returned tensors are grouped
// per field with shape [field][instance][1+candidates], where column 0 is the
// null assignment score and candidate scores are left unscaled.
func (h RecordHead) AssignLogits(instStates, fieldQueryStates [][]float32, fieldCandidateStates [][][]float32) ([][][]float32, error) {
	if err := h.Validate(); err != nil {
		return nil, err
	}
	if len(fieldQueryStates) != len(fieldCandidateStates) {
		return nil, fmt.Errorf("record head field candidates len=%d want=%d", len(fieldCandidateStates), len(fieldQueryStates))
	}
	instQ, err := projectRows(h.InstProjection, instStates, "record head instance states")
	if err != nil {
		return nil, err
	}
	fieldQ, err := projectRows(h.FieldProjection, fieldQueryStates, "record head field query states")
	if err != nil {
		return nil, err
	}
	out := make([][][]float32, len(fieldCandidateStates))
	query := make([]float32, h.RecordDim)
	for f, cand := range fieldCandidateStates {
		rows := makeMatrix(len(instStates), 1+len(cand))
		for i := range instQ {
			for d := 0; d < h.RecordDim; d++ {
				query[d] = instQ[i][d] + fieldQ[f][d]
			}
			rows[i][0] = simd.Sdot(query, h.NullEmbed)
		}
		if len(cand) == 0 {
			out[f] = rows
			continue
		}
		candP, err := projectRows(h.CandProjection, cand, fmt.Sprintf("record head field %d candidate states", f))
		if err != nil {
			return nil, err
		}
		for i := range instQ {
			for d := 0; d < h.RecordDim; d++ {
				query[d] = instQ[i][d] + fieldQ[f][d]
			}
			for c := range candP {
				rows[i][c+1] = simd.Sdot(query, candP[c])
			}
		}
		out[f] = rows
	}
	return out, nil
}

// AnchorlessStates mirrors RecordHead._anchorless_states. It starts from the
// learned instance embeddings [I,H], attends across all non-empty field
// candidates, and returns the residual-pooled states. If there are no
// candidates it returns a deep copy of the learned embeddings.
func (h RecordHead) AnchorlessStates(fieldCandidateStates [][][]float32) ([][]float32, error) {
	if err := h.Validate(); err != nil {
		return nil, err
	}
	inst := h.learnedInstanceStates()
	ctx := make([][]float32, 0)
	for f, cand := range fieldCandidateStates {
		for i, row := range cand {
			if len(row) != h.HiddenSize {
				return nil, fmt.Errorf("record head field %d candidate_states[%d] len=%d want=%d", f, i, len(row), h.HiddenSize)
			}
			ctx = append(ctx, row)
		}
	}
	if len(ctx) == 0 {
		return inst, nil
	}
	q, err := projectRows(h.QProjection, inst, "record head learned instances")
	if err != nil {
		return nil, err
	}
	k, err := projectRows(h.KProjection, ctx, "record head anchorless context")
	if err != nil {
		return nil, err
	}
	v, err := projectRows(h.VProjection, ctx, "record head anchorless context")
	if err != nil {
		return nil, err
	}
	out := makeMatrix(h.InstanceQueries, h.HiddenSize)
	scores := make([]float32, len(ctx))
	scale := float32(1 / math.Sqrt(float64(h.RecordDim)))
	for i := range inst {
		copy(out[i], inst[i])
		for j := range ctx {
			scores[j] = simd.Sdot(q[i], k[j]) * scale
		}
		if !simd.SoftmaxInPlace(scores) {
			return nil, fmt.Errorf("record head anchorless softmax rejected ctx=%d", len(ctx))
		}
		for j, weight := range scores {
			if weight == 0 {
				continue
			}
			for d := 0; d < h.HiddenSize; d++ {
				out[i][d] += weight * v[j][d]
			}
		}
	}
	return out, nil
}

func (h RecordHead) learnedInstanceStates() [][]float32 {
	flat := append([]float32(nil), h.InstanceEmbed...)
	return rowsFromFlat(flat, h.InstanceQueries, h.HiddenSize)
}
