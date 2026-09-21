package gliner2

import "fmt"

func validateTextSchemaRecordMetadata(s TextSchema) error {
	if s.Record == nil {
		return nil
	}
	if s.Marker != "[C]" {
		return fmt.Errorf("record metadata requires [C] marker")
	}
	if err := s.Record.Validate(); err != nil {
		return err
	}
	if len(s.Record.Fields) != len(s.Labels) {
		return fmt.Errorf("record metadata fields=%d want labels=%d", len(s.Record.Fields), len(s.Labels))
	}
	seen := make([]bool, len(s.Labels))
	for _, field := range s.Record.Fields {
		if field.QueryID < 0 || field.QueryID >= len(s.Labels) {
			return fmt.Errorf("record field %q query_id=%d outside [0,%d)", field.Name, field.QueryID, len(s.Labels))
		}
		if seen[field.QueryID] {
			return fmt.Errorf("duplicate record query_id=%d", field.QueryID)
		}
		if want := s.Labels[field.QueryID]; field.Name != want {
			return fmt.Errorf("record field query_id=%d name=%q want %q", field.QueryID, field.Name, want)
		}
		seen[field.QueryID] = true
	}
	for i, ok := range seen {
		if !ok {
			return fmt.Errorf("record metadata missing query_id=%d label=%q", i, s.Labels[i])
		}
	}
	return nil
}

func globalRecordSpecForSchema(schema TextSchema, globalQueryIDs []int) (RecordSpec, error) {
	if schema.Record == nil {
		return RecordSpec{}, fmt.Errorf("record metadata required")
	}
	if err := validateTextSchemaRecordMetadata(schema); err != nil {
		return RecordSpec{}, err
	}
	if len(globalQueryIDs) != len(schema.Labels) {
		return RecordSpec{}, fmt.Errorf("record group query count=%d want=%d", len(globalQueryIDs), len(schema.Labels))
	}
	seen := make(map[int]bool, len(globalQueryIDs))
	for _, qid := range globalQueryIDs {
		if qid < 0 || seen[qid] {
			return RecordSpec{}, fmt.Errorf("invalid or duplicate record query ID")
		}
		seen[qid] = true
	}
	spec := cloneRecordSpec(*schema.Record)
	for i := range spec.Fields {
		spec.Fields[i].QueryID = globalQueryIDs[spec.Fields[i].QueryID]
	}
	if spec.Mode == RecordModeNatural {
		spec.AnchorQueryID = globalQueryIDs[spec.AnchorQueryID]
	}
	return spec, nil
}

func (m *EntityModel) projectRecordCandidateStates(boundaryStates [][]float32, pool PooledCandidates) ([][]float32, error) {
	if m == nil || m.CandidateEncoder == nil {
		return nil, fmt.Errorf("record candidate encoder unavailable")
	}
	if len(pool.ValidMask) != len(pool.Indices) {
		return nil, fmt.Errorf("record candidate valid_mask len=%d want=%d", len(pool.ValidMask), len(pool.Indices))
	}
	endpoints := make([][]float32, len(pool.Indices))
	for i, span := range pool.Indices {
		if len(span) != 2 {
			return nil, fmt.Errorf("record candidate %d boundaries len=%d want=2", i, len(span))
		}
		if span[0] < 0 || span[0] >= len(boundaryStates) || span[1] < 0 || span[1] >= len(boundaryStates) {
			return nil, fmt.Errorf("record candidate %d boundary outside states", i)
		}
		endpoints[i] = append(append([]float32(nil), boundaryStates[span[0]]...), boundaryStates[span[1]]...)
	}
	states, err := projectRows(*m.CandidateEncoder, endpoints, "record candidate endpoints")
	if err != nil {
		return nil, err
	}
	for i, valid := range pool.ValidMask {
		if !valid {
			clear(states[i])
		}
	}
	return states, nil
}
