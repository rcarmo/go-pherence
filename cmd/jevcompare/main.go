package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/rcarmo/go-pherence/model/decider"
	"github.com/rcarmo/go-pherence/model/laya"
	"github.com/rcarmo/go-pherence/model/modernbert"
	"github.com/rcarmo/go-pherence/model/nimble"
	"github.com/rcarmo/go-pherence/model/openjev"
)

type candidate struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}
type request struct {
	Evidence    string      `json:"evidence"`
	Question    string      `json:"question"`
	Candidates  []candidate `json:"candidates"`
	Temperature float64     `json:"temperature"`
}
type input struct {
	ID      string  `json:"id"`
	Task    string  `json:"task"`
	Variant string  `json:"variant"`
	GoldID  string  `json:"gold_id"`
	Request request `json:"request"`
}
type optionScore struct {
	ID            string    `json:"id"`
	Text          string    `json:"text"`
	Score         float32   `json:"score"`
	Logits        []float32 `json:"logits,omitempty"`
	Probabilities []float32 `json:"probabilities,omitempty"`
}
type output struct {
	Version         int           `json:"version"`
	Arm             string        `json:"arm"`
	ID              string        `json:"id"`
	Task            string        `json:"task"`
	Variant         string        `json:"variant"`
	GoldID          string        `json:"gold_id"`
	ModelID         string        `json:"model_id"`
	DecisionSeconds float64       `json:"decision_seconds"`
	SelectedID      string        `json:"selected_id,omitempty"`
	Options         []optionScore `json:"options,omitempty"`
	InputTokens     int           `json:"input_tokens,omitempty"`
	Error           string        `json:"error,omitempty"`
}
type selection struct {
	Scope   string `json:"scope"`
	Cohorts map[string]struct {
		Path   string `json:"path"`
		Rows   int    `json:"rows"`
		SHA256 string `json:"sha256"`
	} `json:"cohorts"`
}

type scorer interface {
	Score(input) (string, []optionScore, int, error)
	Close() error
	ModelID() string
}
type openJEVScorer struct{ r *openjev.Runtime }
type deciderScorer struct{ r *decider.Runtime }
type layaScorer struct {
	m   *laya.Model
	tok *modernbert.Tokenizer
	cfg laya.Config
}
type nimbleScorer struct{ r *nimble.Runtime }

func digest(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func stateText(in input) string {
	if strings.TrimSpace(in.Request.Evidence) == "" {
		return "No separate evidence supplied."
	}
	return "Evidence:\n" + in.Request.Evidence
}
func questionText(in input) string { return stateText(in) + "\nQuestion:\n" + in.Request.Question }
func validate(in input) error {
	if in.ID == "" || in.Task == "" || in.Variant == "" || in.GoldID == "" || strings.TrimSpace(in.Request.Question) == "" || in.Request.Temperature != 1 || len(in.Request.Candidates) < 2 {
		return errors.New("invalid request")
	}
	seen := map[string]bool{}
	gold := false
	for _, c := range in.Request.Candidates {
		if c.ID == "" || strings.TrimSpace(c.Text) == "" || seen[c.ID] {
			return errors.New("invalid candidate")
		}
		seen[c.ID] = true
		gold = gold || c.ID == in.GoldID
	}
	if !gold {
		return errors.New("gold candidate absent")
	}
	return nil
}
func choiceIndex(in input, id string) int {
	for i, c := range in.Request.Candidates {
		if c.ID == id {
			return i
		}
	}
	return -1
}
func readInputs(payload []byte) ([]input, error) {
	var rows []input
	scan := bufio.NewScanner(bytes.NewReader(payload))
	scan.Buffer(make([]byte, 64<<10), 2<<20)
	seen := map[string]bool{}
	for scan.Scan() {
		var in input
		d := json.NewDecoder(bytes.NewReader(scan.Bytes()))
		d.DisallowUnknownFields()
		if err := d.Decode(&in); err != nil {
			return nil, err
		}
		if d.Decode(new(any)) != io.EOF {
			return nil, errors.New("trailing request data")
		}
		if err := validate(in); err != nil {
			return nil, fmt.Errorf("row %d: %w", len(rows)+1, err)
		}
		key := in.Task + "\x00" + in.ID + "\x00" + in.Variant
		if seen[key] {
			return nil, errors.New("duplicate request")
		}
		seen[key] = true
		rows = append(rows, in)
	}
	return rows, scan.Err()
}
func validateOutput(o output, in input, arm, modelID string) error {
	if o.Version != 1 || o.Arm != arm || o.ID != in.ID || o.Task != in.Task || o.Variant != in.Variant || o.GoldID != in.GoldID || o.ModelID != modelID || o.DecisionSeconds < 0 || math.IsNaN(o.DecisionSeconds) || math.IsInf(o.DecisionSeconds, 0) {
		return errors.New("output identity or timing mismatch")
	}
	if (o.Error == "") == (o.SelectedID == "") {
		return errors.New("output must contain exactly one result or error")
	}
	if o.Error != "" {
		if len(o.Options) != 0 {
			return errors.New("rejected output contains option results")
		}
		return nil
	}
	if choiceIndex(in, o.SelectedID) < 0 || len(o.Options) != len(in.Request.Candidates) {
		return errors.New("output candidate shape mismatch")
	}
	for i, candidate := range in.Request.Candidates {
		got := o.Options[i]
		if got.ID != candidate.ID || got.Text != candidate.Text || math.IsNaN(float64(got.Score)) || math.IsInf(float64(got.Score), 0) {
			return errors.New("output candidate mismatch")
		}
		for _, values := range [][]float32{got.Logits, got.Probabilities} {
			for _, value := range values {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					return errors.New("non-finite output")
				}
			}
		}
	}
	return nil
}
func readResume(path string, inputs []input, arm, modelID string) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, err
	}
	defer f.Close()
	scan := bufio.NewScanner(f)
	scan.Buffer(make([]byte, 64<<10), 2<<20)
	count := 0
	for scan.Scan() {
		if count >= len(inputs) {
			return 0, errors.New("resume output has excess rows")
		}
		var out output
		d := json.NewDecoder(bytes.NewReader(scan.Bytes()))
		d.DisallowUnknownFields()
		if err := d.Decode(&out); err != nil {
			return 0, fmt.Errorf("resume row %d: %w", count+1, err)
		}
		if d.Decode(new(any)) != io.EOF {
			return 0, fmt.Errorf("resume row %d has trailing data", count+1)
		}
		if err := validateOutput(out, inputs[count], arm, modelID); err != nil {
			return 0, fmt.Errorf("resume row %d: %w", count+1, err)
		}
		count++
	}
	return count, scan.Err()
}

func (s *openJEVScorer) ModelID() string { return openjev.ModelID + "@" + openjev.ModelPin }
func (s *openJEVScorer) Close() error    { return s.r.Close() }
func (s *openJEVScorer) Score(in input) (string, []optionScore, int, error) {
	opts := make([]optionScore, len(in.Request.Candidates))
	best := 0
	tokens := 0
	premise := questionText(in)
	for i, c := range in.Request.Candidates {
		p, e := s.r.Score(premise, "The correct answer is: "+c.Text)
		if e != nil {
			return "", nil, tokens, e
		}
		tokens += len(p.TokenIDs)
		opts[i] = optionScore{ID: c.ID, Text: c.Text, Score: p.Probabilities[1], Logits: p.Logits, Probabilities: p.Probabilities}
		if opts[i].Score > opts[best].Score {
			best = i
		}
	}
	return opts[best].ID, opts, tokens, nil
}
func (s *deciderScorer) ModelID() string { return decider.ModelID + "@" + decider.ModelPin }
func (s *deciderScorer) Close() error    { return s.r.Close() }
func (s *deciderScorer) Score(in input) (string, []optionScore, int, error) {
	criteria := make([]decider.Criterion, len(in.Request.Candidates))
	for i, c := range in.Request.Candidates {
		criteria[i] = decider.Criterion{Name: c.ID, Description: c.Text}
	}
	r, e := s.r.SystemOne(stateText(in), []decider.NamedQuestion{{ID: "choice", Question: decider.Question{Type: decider.Choice, Instructions: in.Request.Question, Criteria: criteria}}})
	if e != nil {
		return "", nil, 0, e
	}
	if len(r.Answers) != 1 {
		return "", nil, r.Usage.InputTokens, errors.New("decider result shape")
	}
	a := r.Answers[0].Answer
	out := make([]optionScore, len(criteria))
	for i, c := range in.Request.Candidates {
		out[i] = optionScore{ID: c.ID, Text: c.Text, Score: a.Probabilities[c.ID]}
	}
	return a.Choice, out, r.Usage.InputTokens, nil
}
func (s *layaScorer) ModelID() string {
	return "convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982"
}
func (s *layaScorer) Close() error { return nil }
func (s *layaScorer) Score(in input) (string, []optionScore, int, error) {
	criteria := make([]laya.Criterion, len(in.Request.Candidates))
	for i, c := range in.Request.Candidates {
		criteria[i] = laya.Criterion{ID: c.ID, Description: c.Text}
	}
	q := laya.Question{Type: laya.Choice, Instructions: in.Request.Question, Criteria: criteria}
	ids, _, e := laya.BuildSequence(s.tok, stateText(in), q, s.cfg.MaxLen, s.cfg.HeadMaxLen, false)
	if e != nil {
		return "", nil, 0, e
	}
	r, e := s.m.SystemOne(s.tok, stateText(in), []laya.NamedQuestion{{ID: "choice", Question: q}}, s.cfg)
	if e != nil {
		return "", nil, len(ids), e
	}
	a := r.Answers["choice"]
	out := make([]optionScore, len(criteria))
	for i, c := range in.Request.Candidates {
		out[i] = optionScore{ID: c.ID, Text: c.Text, Score: a.Probabilities[c.ID]}
	}
	return a.Choice, out, len(ids), nil
}
func (s *nimbleScorer) ModelID() string { return "bespokelabs/Bespoke-Nimble-9B@" + nimble.ModelPin }
func (s *nimbleScorer) Close() error    { return s.r.Close() }
func (s *nimbleScorer) Score(in input) (string, []optionScore, int, error) {
	choices := make([]any, len(in.Request.Candidates))
	desc := map[string]string{}
	for i, c := range in.Request.Candidates {
		choices[i] = c.ID
		desc[c.ID] = c.Text
	}
	f := nimble.Field{Name: "choice", Type: "enum", Description: in.Request.Question, Choices: choices, ChoiceDescriptions: desc}
	prompts, e := nimble.RenderPrompts(questionText(in), []nimble.Field{f})
	if e != nil {
		return "", nil, 0, e
	}
	ids, _, e := s.r.Tokenize(prompts)
	if e != nil {
		return "", nil, 0, e
	}
	r, e := s.r.Score(questionText(in), []nimble.Field{f}, 1)
	if e != nil {
		return "", nil, len(ids[0]), e
	}
	fr := r.Fields["choice"]
	out := make([]optionScore, len(in.Request.Candidates))
	for i, c := range in.Request.Candidates {
		out[i] = optionScore{ID: c.ID, Text: c.Text, Score: fr.Scores[c.ID], Logits: []float32{fr.Logits[c.ID]}}
	}
	selected, _ := fr.Value.(string)
	return selected, out, len(ids[0]), nil
}

func loadScorer(arm, model string, maxInput int) (scorer, error) {
	switch arm {
	case "openjev":
		r, e := openjev.Load(model, maxInput)
		if e != nil {
			return nil, e
		}
		return &openJEVScorer{r}, nil
	case "decider":
		r, e := decider.Load(model, maxInput)
		if e != nil {
			return nil, e
		}
		return &deciderScorer{r}, nil
	case "laya":
		m, c, e := laya.Load(model, filepath.Join(model, "encoder", "config.json"))
		if e != nil {
			return nil, e
		}
		tok, e := modernbert.LoadTokenizer(filepath.Join(model, "tokenizer"))
		if e != nil {
			return nil, e
		}
		return &layaScorer{m, tok, c}, nil
	case "nimble":
		r, e := nimble.LoadMerged(model, maxInput)
		if e != nil {
			return nil, e
		}
		return &nimbleScorer{r}, nil
	default:
		return nil, errors.New("arm must be openjev, decider, laya or nimble")
	}
}
func main() {
	if e := run(); e != nil {
		fmt.Fprintln(os.Stderr, "jevcompare:", e)
		os.Exit(1)
	}
}
func run() error {
	arm := flag.String("arm", "", "candidate arm")
	study := flag.String("study", "", "frozen study directory")
	cohort := flag.String("cohort", "screening", "screening or finalist")
	model := flag.String("model", "", "local released model directory")
	dest := flag.String("output", "", "new JSONL output")
	resume := flag.Bool("resume", false, "validate and append to an existing strict output prefix")
	cpuProfile := flag.String("cpuprofile", "", "new Go CPU profile covering scoring only")
	maxInput := flag.Int("max-input", 4096, "model input token limit")
	flag.Parse()
	if *arm == "" || *study == "" || *model == "" || *dest == "" || (*cohort != "screening" && *cohort != "finalist") {
		return errors.New("arm/study/cohort/model/output required")
	}
	b, e := os.ReadFile(filepath.Join(*study, "selection.json"))
	if e != nil {
		return e
	}
	var sel selection
	if e = json.Unmarshal(b, &sel); e != nil {
		return e
	}
	if sel.Scope != "jev-port-bakeoff-v1" {
		return errors.New("wrong study scope")
	}
	c, ok := sel.Cohorts[*cohort]
	if !ok {
		return errors.New("cohort absent")
	}
	payload, e := os.ReadFile(filepath.Join(*study, c.Path))
	if e != nil {
		return e
	}
	if digest(payload) != c.SHA256 {
		return errors.New("cohort hash mismatch")
	}
	inputs, e := readInputs(payload)
	if e != nil {
		return e
	}
	if len(inputs) != c.Rows {
		return fmt.Errorf("cohort row count %d != %d", len(inputs), c.Rows)
	}
	s, e := loadScorer(*arm, *model, *maxInput)
	if e != nil {
		return e
	}
	defer s.Close()
	var profile *os.File
	if *cpuProfile != "" {
		profile, e = os.OpenFile(*cpuProfile, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
		if e != nil {
			return e
		}
		if e = pprof.StartCPUProfile(profile); e != nil {
			profile.Close()
			return e
		}
		defer func() {
			pprof.StopCPUProfile()
			_ = profile.Close()
		}()
	}
	count := 0
	flags := os.O_WRONLY | os.O_CREATE | os.O_EXCL
	if *resume {
		count, e = readResume(*dest, inputs, *arm, s.ModelID())
		if e != nil {
			return e
		}
		flags = os.O_WRONLY | os.O_CREATE | os.O_APPEND
	}
	f, e := os.OpenFile(*dest, flags, 0o644)
	if e != nil {
		return e
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	resumed := count
	for ; count < len(inputs); count++ {
		in := inputs[count]
		o := output{Version: 1, Arm: *arm, ID: in.ID, Task: in.Task, Variant: in.Variant, GoldID: in.GoldID, ModelID: s.ModelID()}
		start := time.Now()
		o.SelectedID, o.Options, o.InputTokens, e = s.Score(in)
		o.DecisionSeconds = time.Since(start).Seconds()
		if e != nil {
			o.Error = e.Error()
		} else if choiceIndex(in, o.SelectedID) < 0 {
			o.Error = "selected candidate absent"
			o.SelectedID = ""
		}
		if e = validateOutput(o, in, *arm, s.ModelID()); e != nil {
			return fmt.Errorf("row %d: %w", count+1, e)
		}
		if e = enc.Encode(o); e != nil {
			return e
		}
		if e = f.Sync(); e != nil {
			return e
		}
		fmt.Fprintf(os.Stderr, "arm=%s row=%d/%d id=%s variant=%s seconds=%.3f rejected=%v\n", *arm, count+1, c.Rows, in.ID, in.Variant, o.DecisionSeconds, o.Error != "")
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"complete": true, "arm": *arm, "cohort": *cohort, "rows": count, "resumed": resumed, "model_id": s.ModelID(), "output": *dest})
}
