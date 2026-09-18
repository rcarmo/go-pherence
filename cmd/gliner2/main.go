// gliner2 runs native entity scoring and explicit basic span decoding.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strings"

	"github.com/rcarmo/go-pherence/model/gliner2"
)

type labelList []string

func (l *labelList) String() string     { return fmt.Sprint([]string(*l)) }
func (l *labelList) Set(s string) error { *l = append(*l, s); return nil }
func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run(args []string, out, stderr io.Writer) error {
	fs := flag.NewFlagSet("gliner2", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("model", "", "local GLiNER2.5 checkpoint directory")
	text := fs.String("text", "", "input text")
	schemaPath := fs.String("schema", "", "ordered TextSchema JSON file for entities or classification")
	schemasPath := fs.String("schemas", "", "JSON array of entity/classification/relation schemas sharing one encoder pass")
	classify := fs.String("classify", "", "classification task name; returns independent choice scores instead of entities")
	record := fs.String("record", "", "single record schema name")
	mode := fs.String("record-mode", "natural", "natural, latent or anchorless")
	anchor := fs.String("anchor", "", "natural anchor field; defaults to first field")
	var fields labelList
	fs.Var(&fields, "field", "ordered field name:str|list[,required][,exclusive]; repeat")
	relation := fs.String("relation", "", "single relation type; infers head and tail roles")
	threshold := fs.Float64("threshold", .5, "sigmoid threshold before checkpoint count/abstention filtering")
	policy := fs.String("overlap", "flat", "allow, nested, flat, or longest (per label)")
	maxTokens := fs.Int("max-tokens", 4096, "reject longer formatted subword sequences")
	raw := fs.Bool("raw", false, "emit raw candidate scores instead of decoded entities")
	var labels labelList
	fs.Var(&labels, "label", "entity label; repeat in schema order")
	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return nil
		}
		return err
	}
	if fs.NArg() != 0 || *dir == "" || *text == "" || (len(labels) == 0 && *relation == "" && *record == "" && *schemaPath == "" && *schemasPath == "") {
		return fmt.Errorf("-model, -text and labels or a task/schema are required; positional arguments not accepted")
	}
	var schemas []gliner2.TextSchema
	if *schemasPath != "" {
		if *schemaPath != "" || *relation != "" || *record != "" || *classify != "" || len(labels) > 0 || len(fields) > 0 || *anchor != "" {
			return fmt.Errorf("-schemas cannot be combined with other task/schema flags")
		}
		var err error
		schemas, err = readTextSchemas(*schemasPath)
		if err != nil {
			return err
		}
	}
	var schema *gliner2.TextSchema
	if *schemaPath != "" {
		if *relation != "" || *record != "" || *classify != "" || len(labels) > 0 || len(fields) > 0 || *anchor != "" {
			return fmt.Errorf("-schema cannot be combined with task/label/field flags")
		}
		loaded, err := readTextSchema(*schemaPath)
		if err != nil {
			return err
		}
		schema = &loaded
	}
	if *relation != "" && (*classify != "" || len(labels) != 0) {
		return fmt.Errorf("-relation cannot be combined with -classify or -label")
	}
	if *record == "" && (len(fields) > 0 || *anchor != "") {
		return fmt.Errorf("-field and -anchor require -record")
	}
	if *record != "" && (*relation != "" || *classify != "" || len(labels) > 0) {
		return fmt.Errorf("-record cannot be combined with other task modes")
	}
	if math.IsNaN(*threshold) || *threshold < 0 || *threshold > 1 || *maxTokens <= 0 {
		return fmt.Errorf("invalid threshold or token budget")
	}
	if *record != "" {
		spec, err := recordSpec(fields, *mode, *anchor)
		if err != nil {
			return err
		}
		m, err := gliner2.LoadRecordModel(*dir)
		if err != nil {
			return err
		}
		scores, err := m.ScoreRecord(*text, *record, spec, *maxTokens)
		if err != nil {
			return err
		}
		enc := json.NewEncoder(out)
		enc.SetIndent("", "  ")
		if *raw {
			return enc.Encode(scores)
		}
		cfg := m.Base.Config.BoundaryHead
		cfg.RecordAnchorThreshold = *threshold
		cfg.OverlapPolicy = *policy
		decoded, err := gliner2.DecodeRecords(*text, scores, cfg)
		if err != nil {
			return err
		}
		return enc.Encode(struct {
			Records []gliner2.DecodedRecord `json:"records"`
		}{decoded})
	}
	m, err := gliner2.LoadEntityModel(*dir)
	if err != nil {
		return err
	}
	if len(schemas) > 0 {
		scores, err := m.ScoreSchemas(*text, schemas, *maxTokens)
		if err != nil {
			return err
		}
		enc := json.NewEncoder(out)
		enc.SetIndent("", "  ")
		if *raw {
			return enc.Encode(scores)
		}
		groups, err := gliner2.DecodeSchemaGroups(*text, scores, *threshold, *policy, m.Config.BoundaryHead)
		if err != nil {
			return err
		}
		return enc.Encode(struct {
			Groups []gliner2.DecodedSchemaGroup `json:"groups"`
		}{groups})
	}
	if schema != nil {
		enc := json.NewEncoder(out)
		enc.SetIndent("", "  ")
		if schema.Marker == "[L]" {
			result, err := m.ClassifySchema(*text, *schema, *maxTokens)
			if err != nil {
				return err
			}
			return enc.Encode(result)
		}
		scores, err := m.ScoreEntitySchema(*text, *schema, *maxTokens)
		if err != nil {
			return err
		}
		if *raw {
			return enc.Encode(scores)
		}
		entities, err := gliner2.DecodeConfiguredEntities(*text, scores, *threshold, *policy, m.Config.BoundaryHead)
		if err != nil {
			return err
		}
		return enc.Encode(struct {
			Entities []gliner2.Entity `json:"entities"`
		}{entities})
	}
	if *relation != "" {
		result, err := m.ScoreRelation(*text, *relation, *maxTokens)
		if err != nil {
			return err
		}
		enc := json.NewEncoder(out)
		enc.SetIndent("", "  ")
		if *raw {
			return enc.Encode(result)
		}
		decoded, err := gliner2.DecodeRelations(*text, result, *threshold, m.Config.BoundaryHead.RelationTemperature)
		if err != nil {
			return err
		}
		return enc.Encode(struct {
			Relations []gliner2.Relation `json:"relations"`
		}{decoded})
	}
	if *classify != "" {
		result, err := m.Classify(*text, *classify, labels, *maxTokens)
		if err != nil {
			return err
		}
		enc := json.NewEncoder(out)
		enc.SetIndent("", "  ")
		return enc.Encode(result)
	}
	scores, err := m.ScoreEntities(*text, labels, *maxTokens)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	if *raw {
		return enc.Encode(scores)
	}
	entities, err := gliner2.DecodeConfiguredEntities(*text, scores, *threshold, *policy, m.Config.BoundaryHead)
	if err != nil {
		return err
	}
	return enc.Encode(struct {
		Entities []gliner2.Entity `json:"entities"`
		Decoding string           `json:"decoding"`
	}{entities, "checkpoint temperature/count/abstention with per-label overlap policy"})
}

func recordSpec(fields []string, mode, anchor string) (gliner2.RecordSpec, error) {
	s := gliner2.RecordSpec{Mode: mode, AnchorQueryID: -1}
	seen := map[string]bool{}
	for i, field := range fields {
		name, definition, ok := strings.Cut(field, ":")
		parts := strings.Split(definition, ",")
		kind := parts[0]
		if !ok || strings.TrimSpace(name) == "" || seen[name] || (kind != "str" && kind != "list") {
			return s, fmt.Errorf("field must be a unique name:str or name:list: %q", field)
		}
		seen[name] = true
		f := gliner2.RecordField{QueryID: i, Name: name, Scalar: kind == "str"}
		modifiers := map[string]bool{}
		for _, modifier := range parts[1:] {
			if modifiers[modifier] {
				return s, fmt.Errorf("duplicate field modifier %q", modifier)
			}
			modifiers[modifier] = true
			switch modifier {
			case "required":
				f.Required = true
			case "exclusive":
				f.Exclusive = true
			default:
				return s, fmt.Errorf("unknown field modifier %q", modifier)
			}
		}
		s.Fields = append(s.Fields, f)
		if name == anchor {
			s.AnchorQueryID = i
		}
	}
	if mode == gliner2.RecordModeNatural {
		if anchor == "" && len(s.Fields) > 0 {
			s.AnchorQueryID = 0
		}
		if s.AnchorQueryID < 0 {
			return s, fmt.Errorf("natural anchor must name a field")
		}
	} else if anchor != "" {
		return s, fmt.Errorf("only natural mode accepts -anchor")
	}
	return s, s.Validate()
}

func readTextSchema(path string) (gliner2.TextSchema, error) {
	var s gliner2.TextSchema
	f, err := os.Open(path)
	if err != nil {
		return s, err
	}
	defer f.Close()
	d := json.NewDecoder(f)
	d.DisallowUnknownFields()
	if err = d.Decode(&s); err != nil {
		return s, err
	}
	var extra any
	if err = d.Decode(&extra); err != io.EOF {
		return s, fmt.Errorf("schema contains trailing data")
	}
	return s, validateTextSchema(s)
}

func validateTextSchema(s gliner2.TextSchema) error {
	if (s.Marker != "[E]" && s.Marker != "[L]") || strings.TrimSpace(s.Parent) == "" || len(s.Labels) == 0 {
		return fmt.Errorf("schema needs parent, labels and [E] or [L] marker")
	}
	seen := map[string]bool{}
	for _, label := range s.Labels {
		if strings.TrimSpace(label) == "" || seen[label] {
			return fmt.Errorf("empty/duplicate schema label")
		}
		seen[label] = true
	}
	return nil
}

func readTextSchemas(path string) ([]gliner2.TextSchema, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	d := json.NewDecoder(f)
	d.DisallowUnknownFields()
	var schemas []gliner2.TextSchema
	if err = d.Decode(&schemas); err != nil {
		return nil, err
	}
	var extra any
	if err = d.Decode(&extra); err != io.EOF {
		return nil, fmt.Errorf("schemas contain trailing data")
	}
	if len(schemas) == 0 {
		return nil, fmt.Errorf("at least one schema required")
	}
	for i, s := range schemas {
		check := s
		if s.Marker == "[R]" {
			if len(s.Labels) != 2 || s.Labels[0] != "head" || s.Labels[1] != "tail" {
				return nil, fmt.Errorf("schema %d: relations require ordered head/tail roles", i)
			}
			check.Marker = "[E]" // Reuse parent/label validation, not task dispatch.
		}
		if err := validateTextSchema(check); err != nil {
			return nil, fmt.Errorf("schema %d: %w", i, err)
		}
	}
	return schemas, nil
}
