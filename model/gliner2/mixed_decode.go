package gliner2

import "fmt"

// DecodedSchemaGroup retains group order even when labels repeat across tasks.
type DecodedSchemaGroup struct {
	Parent         string                `json:"parent"`
	Marker         string                `json:"marker"`
	Entities       []Entity              `json:"entities,omitempty"`
	Classification *ClassificationScores `json:"classification,omitempty"`
}

// DecodeSchemaGroups currently decodes entity and classification groups. It
// rejects raw relation/record role groups rather than mislabelling them entities.
func DecodeSchemaGroups(text string, s MixedScores, threshold float64, policy string, c BoundaryHeadConfig) ([]DecodedSchemaGroup, error) {
	if len(s.GroupQueryIDs) != len(s.Input.Groups) {
		return nil, fmt.Errorf("group routing count mismatch")
	}
	result := make([]DecodedSchemaGroup, len(s.Input.Groups))
	for i, g := range s.Input.Groups {
		result[i].Parent = g.Schema.Parent
		result[i].Marker = g.Schema.Marker
		switch g.Schema.Marker {
		case "[L]":
			cls, ok := s.Classifications[i]
			if !ok {
				return nil, fmt.Errorf("missing classification group %d", i)
			}
			if len(cls.Labels) != len(g.Schema.Labels) || len(cls.Logits) != len(cls.Labels) || len(cls.Probabilities) != len(cls.Labels) {
				return nil, fmt.Errorf("classification group shape mismatch")
			}
			result[i].Classification = &cls
		case "[E]":
			if s.Extraction == nil {
				return nil, fmt.Errorf("missing extractive scores")
			}
			ids := s.GroupQueryIDs[i]
			if len(ids) != len(g.Schema.Labels) {
				return nil, fmt.Errorf("entity group query count mismatch")
			}
			e := *s.Extraction
			e.Input.Labels = append([]string(nil), g.Schema.Labels...)
			e.Logits = make([][]float32, len(s.Extraction.Logits))
			e.NullLogits = nil
			e.CountLogits = nil
			seen := map[int]bool{}
			for j, q := range ids {
				if q < 0 || seen[q] {
					return nil, fmt.Errorf("invalid or duplicate query ID")
				}
				seen[q] = true
				for k, row := range s.Extraction.Logits {
					if q >= len(row) {
						return nil, fmt.Errorf("query outside score axis")
					}
					if j == 0 {
						e.Logits[k] = make([]float32, len(ids))
					}
					e.Logits[k][j] = row[q]
				}
				if c.EnableAbstention {
					if q >= len(s.Extraction.NullLogits) {
						return nil, fmt.Errorf("query outside null axis")
					}
					e.NullLogits = append(e.NullLogits, s.Extraction.NullLogits[q])
				}
				if c.AdaptiveThreshold {
					if q >= len(s.Extraction.CountLogits) {
						return nil, fmt.Errorf("query outside count axis")
					}
					e.CountLogits = append(e.CountLogits, s.Extraction.CountLogits[q])
				}
			}
			entities, err := DecodeConfiguredEntities(text, e, threshold, policy, c)
			if err != nil {
				return nil, err
			}
			result[i].Entities = entities
		default:
			return nil, fmt.Errorf("mixed decoder does not yet handle marker %q", g.Schema.Marker)
		}
	}
	return result, nil
}
