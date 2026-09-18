package gliner2

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

const (
	RecordModeNatural    = "natural"
	RecordModeLatent     = "latent"
	RecordModeAnchorless = "anchorless"
)

// RecordField binds one record field to its schema query.
type RecordField struct {
	QueryID  int    `json:"query_id"`
	Name     string `json:"name"`
	Scalar   bool   `json:"scalar"`
	Required bool   `json:"required,omitempty"`
}

// RecordSpec is the minimal single-group schema required by dense record
// routing. Natural groups seed instances from one anchor query; latent and
// anchorless groups ignore AnchorQueryID.
type RecordSpec struct {
	Mode          string        `json:"mode"`
	AnchorQueryID int           `json:"anchor_query_id"`
	Fields        []RecordField `json:"fields"`
}

func (s RecordSpec) Validate() error {
	if len(s.Fields) == 0 {
		return fmt.Errorf("record spec requires at least one field")
	}
	for _, f := range s.Fields {
		if f.Required && !f.Scalar {
			return fmt.Errorf("required list fields are not supported yet")
		}
	}
	switch s.Mode {
	case RecordModeNatural:
		if s.anchorFieldIndex() < 0 {
			return fmt.Errorf("record spec natural anchor_query_id=%d missing from fields", s.AnchorQueryID)
		}
	case RecordModeLatent, RecordModeAnchorless:
		return nil
	default:
		return fmt.Errorf("record spec mode=%q", s.Mode)
	}
	return nil
}

func (s RecordSpec) anchorFieldIndex() int {
	for i, field := range s.Fields {
		if field.QueryID == s.AnchorQueryID {
			return i
		}
	}
	return -1
}

// DenseRecordGroupOutput is the shared-pool single-group record representation.
// AssignLogits are instance-major [instance][field][1+candidate]. Column 0 is
// the explicit null assignment alternative.
type DenseRecordGroupOutput struct {
	Spec              RecordSpec    `json:"spec"`
	ObjectLogits      []float32     `json:"object_logits"`
	AssignLogits      [][][]float32 `json:"assign_logits"`
	InstanceMask      []bool        `json:"instance_mask"`
	FieldMembership   [][]bool      `json:"field_membership"`
	PoolSpans         [][]int       `json:"pool_spans"`
	Fields            []RecordField `json:"fields"`
	FieldQueryIDs     []int         `json:"field_query_ids"`
	InstancePoolIndex []int         `json:"instance_pool_index"`
}

// ForwardGroupDense mirrors upstream RecordHead.forward_group_dense for one
// shared candidate pool. It keeps the dense candidate axis intact and masks
// invalid field/candidate membership instead of compacting per-field lists.
func (h RecordHead) ForwardGroupDense(spec RecordSpec, queryStates, candidateStates [][]float32, pool PooledCandidates, pairLogits [][]float32, queryMask []bool) (DenseRecordGroupOutput, error) {
	var out DenseRecordGroupOutput
	if err := h.Validate(); err != nil {
		return out, err
	}
	if err := spec.Validate(); err != nil {
		return out, err
	}
	candidateCount, queryCount, err := validateRecordGroupDenseInputs(h, queryStates, candidateStates, pool, pairLogits, queryMask)
	if err != nil {
		return out, err
	}
	fieldQueryIDs := make([]int, len(spec.Fields))
	for i, field := range spec.Fields {
		fieldQueryIDs[i] = field.QueryID
	}
	safeFieldQueryIDs, validFieldQueryIDs := safeRecordQueryIDs(fieldQueryIDs, queryCount)
	fieldQueries := make([][]float32, len(spec.Fields))
	membership := make([][]bool, len(spec.Fields))
	for f := range spec.Fields {
		fieldQueries[f] = queryStates[safeFieldQueryIDs[f]]
		membership[f] = make([]bool, candidateCount)
		queryValid := validFieldQueryIDs[f] && queryMask[safeFieldQueryIDs[f]]
		for c := 0; c < candidateCount; c++ {
			membership[f][c] = queryValid && pool.ValidMask[c]
		}
	}
	out = DenseRecordGroupOutput{
		Spec:            cloneRecordSpec(spec),
		FieldMembership: membership,
		PoolSpans:       clonePoolSpans(pool.Indices),
		Fields:          append([]RecordField(nil), spec.Fields...),
		FieldQueryIDs:   fieldQueryIDs,
	}

	var instanceStates [][]float32
	switch spec.Mode {
	case RecordModeNatural:
		anchorField := spec.anchorFieldIndex()
		instanceStates = candidateStates
		out.ObjectLogits = gatherPairLogitColumn(pairLogits, safeFieldQueryIDs[anchorField])
		out.InstanceMask = append([]bool(nil), membership[anchorField]...)
		out.InstancePoolIndex = densePoolIndices(candidateCount)
	case RecordModeLatent:
		instanceStates = candidateStates
		out.ObjectLogits, err = h.LatentSeedLogits(candidateStates)
		if err != nil {
			return DenseRecordGroupOutput{}, err
		}
		out.InstanceMask = anyFieldMembership(membership, candidateCount)
		out.InstancePoolIndex = densePoolIndices(candidateCount)
	case RecordModeAnchorless:
		instanceStates, err = h.anchorlessDenseStates(candidateStates, pool.ValidMask)
		if err != nil {
			return DenseRecordGroupOutput{}, err
		}
		out.ObjectLogits, err = h.ObjectLogits(instanceStates)
		if err != nil {
			return DenseRecordGroupOutput{}, err
		}
		out.InstanceMask = make([]bool, h.InstanceQueries)
		out.InstancePoolIndex = make([]int, h.InstanceQueries)
		for i := range out.InstanceMask {
			out.InstanceMask[i] = true
			out.InstancePoolIndex[i] = -1
		}
	default:
		return DenseRecordGroupOutput{}, fmt.Errorf("record spec mode=%q", spec.Mode)
	}
	out.AssignLogits, err = h.assignPoolLogits(instanceStates, fieldQueries, candidateStates, membership)
	if err != nil {
		return DenseRecordGroupOutput{}, err
	}
	return out, nil
}

func validateRecordGroupDenseInputs(h RecordHead, queryStates, candidateStates [][]float32, pool PooledCandidates, pairLogits [][]float32, queryMask []bool) (candidateCount, queryCount int, err error) {
	candidateCount = len(candidateStates)
	if len(pool.Indices) != candidateCount {
		return 0, 0, fmt.Errorf("record group dense pool indices len=%d want=%d", len(pool.Indices), candidateCount)
	}
	if len(pool.ValidMask) != candidateCount {
		return 0, 0, fmt.Errorf("record group dense pool valid_mask len=%d want=%d", len(pool.ValidMask), candidateCount)
	}
	if len(pool.ProposalLogits) != 0 && len(pool.ProposalLogits) != candidateCount {
		return 0, 0, fmt.Errorf("record group dense pool proposal_logits len=%d want=%d", len(pool.ProposalLogits), candidateCount)
	}
	if len(pool.CompatLogits) != 0 && len(pool.CompatLogits) != candidateCount {
		return 0, 0, fmt.Errorf("record group dense pool compat_logits len=%d want=%d", len(pool.CompatLogits), candidateCount)
	}
	if len(pairLogits) != candidateCount {
		return 0, 0, fmt.Errorf("record group dense pair_logits rows=%d want=%d", len(pairLogits), candidateCount)
	}
	if len(queryStates) == 0 || len(queryMask) == 0 {
		return 0, 0, fmt.Errorf("record routing requires at least one boundary query")
	}
	pairQueryCount := len(queryMask)
	if candidateCount > 0 {
		pairQueryCount = len(pairLogits[0])
		for c := range pairLogits {
			if len(pairLogits[c]) != pairQueryCount {
				return 0, 0, fmt.Errorf("record group dense pair_logits[%d] len=%d want=%d", c, len(pairLogits[c]), pairQueryCount)
			}
		}
	}
	queryCount = min(len(queryStates), len(queryMask))
	if pairQueryCount < queryCount {
		queryCount = pairQueryCount
	}
	if queryCount <= 0 {
		return 0, 0, fmt.Errorf("record routing requires at least one boundary query")
	}
	for i := 0; i < queryCount; i++ {
		if len(queryStates[i]) != h.HiddenSize {
			return 0, 0, fmt.Errorf("record group dense query_states[%d] len=%d want=%d", i, len(queryStates[i]), h.HiddenSize)
		}
	}
	for c := range candidateStates {
		if len(candidateStates[c]) != h.HiddenSize {
			return 0, 0, fmt.Errorf("record group dense candidate_states[%d] len=%d want=%d", c, len(candidateStates[c]), h.HiddenSize)
		}
		if len(pool.Indices[c]) != 2 {
			return 0, 0, fmt.Errorf("record group dense pool indices[%d] len=%d want=2", c, len(pool.Indices[c]))
		}
	}
	return candidateCount, queryCount, nil
}

func safeRecordQueryIDs(queryIDs []int, queryCount int) ([]int, []bool) {
	safe := make([]int, len(queryIDs))
	valid := make([]bool, len(queryIDs))
	if queryCount <= 0 {
		return safe, valid
	}
	upper := queryCount - 1
	for i, qid := range queryIDs {
		valid[i] = 0 <= qid && qid <= upper
		safe[i] = qid
		if safe[i] < 0 {
			safe[i] = 0
		}
		if safe[i] > upper {
			safe[i] = upper
		}
	}
	return safe, valid
}

func gatherPairLogitColumn(pairLogits [][]float32, queryID int) []float32 {
	out := make([]float32, len(pairLogits))
	for i := range pairLogits {
		out[i] = pairLogits[i][queryID]
	}
	return out
}

func densePoolIndices(count int) []int {
	out := make([]int, count)
	for i := range out {
		out[i] = i
	}
	return out
}

func anyFieldMembership(membership [][]bool, candidateCount int) []bool {
	out := make([]bool, candidateCount)
	for _, field := range membership {
		for c := 0; c < min(candidateCount, len(field)); c++ {
			out[c] = out[c] || field[c]
		}
	}
	return out
}

func (h RecordHead) anchorlessDenseStates(candidateStates [][]float32, poolMask []bool) ([][]float32, error) {
	if len(poolMask) != len(candidateStates) {
		return nil, fmt.Errorf("record head anchorless dense pool_mask len=%d want=%d", len(poolMask), len(candidateStates))
	}
	inst := h.learnedInstanceStates()
	if len(candidateStates) == 0 {
		return inst, nil
	}
	q, err := projectRows(h.QProjection, inst, "record head anchorless dense instances")
	if err != nil {
		return nil, err
	}
	k, err := projectRows(h.KProjection, candidateStates, "record head anchorless dense candidates")
	if err != nil {
		return nil, err
	}
	v, err := projectRows(h.VProjection, candidateStates, "record head anchorless dense candidates")
	if err != nil {
		return nil, err
	}
	out := makeMatrix(h.InstanceQueries, h.HiddenSize)
	scores := make([]float32, len(candidateStates))
	scale := float32(1 / math.Sqrt(float64(h.RecordDim)))
	for i := range inst {
		copy(out[i], inst[i])
		for c := range candidateStates {
			scores[c] = simd.Sdot(q[i], k[c]) * scale
			if !poolMask[c] {
				scores[c] = MaskLogit
			}
		}
		if !simd.SoftmaxInPlace(scores) {
			return nil, fmt.Errorf("record head anchorless dense softmax rejected candidates=%d", len(candidateStates))
		}
		for c, weight := range scores {
			if weight == 0 {
				continue
			}
			for d := 0; d < h.HiddenSize; d++ {
				out[i][d] += weight * v[c][d]
			}
		}
	}
	return out, nil
}

func (h RecordHead) assignPoolLogits(instanceStates, fieldQueries, candidateStates [][]float32, membership [][]bool) ([][][]float32, error) {
	if len(fieldQueries) != len(membership) {
		return nil, fmt.Errorf("record head dense membership len=%d want=%d", len(membership), len(fieldQueries))
	}
	instQ, err := projectRows(h.InstProjection, instanceStates, "record head dense instance states")
	if err != nil {
		return nil, err
	}
	fieldQ, err := projectRows(h.FieldProjection, fieldQueries, "record head dense field query states")
	if err != nil {
		return nil, err
	}
	candQ, err := projectRows(h.CandProjection, candidateStates, "record head dense candidate states")
	if err != nil {
		return nil, err
	}
	for f := range membership {
		if len(membership[f]) != len(candidateStates) {
			return nil, fmt.Errorf("record head dense membership[%d] len=%d want=%d", f, len(membership[f]), len(candidateStates))
		}
	}
	out := make([][][]float32, len(instanceStates))
	query := make([]float32, h.RecordDim)
	for i := range instQ {
		out[i] = make([][]float32, len(fieldQueries))
		for f := range fieldQ {
			row := make([]float32, 1+len(candidateStates))
			for d := 0; d < h.RecordDim; d++ {
				query[d] = instQ[i][d] + fieldQ[f][d]
			}
			row[0] = simd.Sdot(query, h.NullEmbed)
			for c := range candQ {
				row[c+1] = MaskLogit
				if membership[f][c] {
					row[c+1] = simd.Sdot(query, candQ[c])
				}
			}
			out[i][f] = row
		}
	}
	return out, nil
}

func cloneRecordSpec(spec RecordSpec) RecordSpec {
	spec.Fields = append([]RecordField(nil), spec.Fields...)
	return spec
}

func clonePoolSpans(indices [][]int) [][]int {
	out := make([][]int, len(indices))
	for i, span := range indices {
		out[i] = append([]int(nil), span...)
	}
	return out
}
