// gliner2 runs native entity scoring and explicit basic span decoding.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

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
	classify := fs.String("classify", "", "classification task name; returns independent choice scores instead of entities")
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
	if fs.NArg() != 0 || *dir == "" || *text == "" || (len(labels) == 0 && *relation == "") {
		return fmt.Errorf("-model, -text and at least one -label are required; positional arguments not accepted")
	}
	if *relation != "" && (*classify != "" || len(labels) != 0) {
		return fmt.Errorf("-relation cannot be combined with -classify or -label")
	}
	if *threshold < 0 || *threshold > 1 || *maxTokens <= 0 {
		return fmt.Errorf("invalid threshold or token budget")
	}
	m, err := gliner2.LoadEntityModel(*dir)
	if err != nil {
		return err
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
