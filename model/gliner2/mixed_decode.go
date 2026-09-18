package gliner2

import (
	"fmt"
	"reflect"
)

// DecodedSchemaGroup retains group order even when labels repeat across tasks.
type DecodedSchemaGroup struct {
	Parent         string                `json:"parent"`
	Marker         string                `json:"marker"`
	Relations      []Relation            `json:"relations,omitempty"`
	Entities       []Entity              `json:"entities,omitempty"`
	Records        []DecodedRecord       `json:"records,omitempty"`
	Classification *ClassificationScores `json:"classification,omitempty"`
}

// DecodeSchemaGroups decodes entity, classification, relation and record
// groups. Raw [C] field groups without record metadata are not decoded.
func DecodeSchemaGroups(text string, s MixedScores, threshold float64, policy string, c BoundaryHeadConfig) ([]DecodedSchemaGroup, error) {
	if len(s.GroupQueryIDs) != len(s.Input.Groups) {
		return nil, fmt.Errorf("group routing count mismatch")
	}
	result := make([]DecodedSchemaGroup, len(s.Input.Groups))
	for i, g := range s.Input.Groups {
		result[i].Parent = g.Schema.Parent
		result[i].Marker = g.Schema.Marker
		switch g.Schema.Marker {
		case "[R]":
			r, ok := s.Relations[i]
			if !ok {
				return nil, fmt.Errorf("missing relation group %d", i)
			}
			decoded, err := DecodeRelations(text, r, threshold, c.RelationTemperature)
			if err != nil {
				return nil, err
			}
			result[i].Relations = decoded
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
		case "[C]":
			if g.Schema.Record == nil {
				return nil, fmt.Errorf("record group %d requires record metadata", i)
			}
			r, ok := s.Records[i]
			if !ok {
				return nil, fmt.Errorf("missing record group %d", i)
			}
			expected, err := globalRecordSpecForSchema(g.Schema, s.GroupQueryIDs[i])
			if err != nil {
				return nil, err
			}
			if !reflect.DeepEqual(r.Group.Spec, expected) || !reflect.DeepEqual(r.Group.Fields, expected.Fields) {
				return nil, fmt.Errorf("record group schema mismatch")
			}
			decoded, err := DecodeRecords(text, r, c)
			if err != nil {
				return nil, err
			}
			result[i].Records = decoded
		default:
			return nil, fmt.Errorf("mixed decoder does not yet handle marker %q", g.Schema.Marker)
		}
	}
	return result, nil
}
