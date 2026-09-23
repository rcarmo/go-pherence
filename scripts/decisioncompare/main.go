// decisioncompare records precision trade-offs for the pinned Gemma model.
// It is an opt-in hardware experiment, not an accuracy test or a tolerance gate.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model"
	gso "github.com/rcarmo/go-pherence/model/gosystemone"
)

type cohort struct {
	Description string        `json:"description"`
	Thresholds  []float64     `json:"thresholds"`
	Requests    []gso.Request `json:"requests"`
}
type trial struct {
	Rows         int           `json:"packed_token_rows"`
	Response     gso.Response  `json:"response"`
	Logits       [][][]float32 `json:"node_logits"`
	DeviceBefore string        `json:"device_before"`
	DeviceAfter  string        `json:"device_after"`
}
type errorStats struct {
	Count int     `json:"count"`
	Max   float64 `json:"max_absolute"`
	RMS   float64 `json:"rms"`
	sum   float64
}

func (s *errorStats) add(x float64) {
	s.Count++
	s.Max = math.Max(s.Max, math.Abs(x))
	s.sum += x * x
	s.RMS = math.Sqrt(s.sum / float64(s.Count))
}

type fieldComparison struct {
	Context                int             `json:"context"`
	Field                  string          `json:"field"`
	SerialWinner           json.RawMessage `json:"serial_winner"`
	PackedWinner           json.RawMessage `json:"packed_winner"`
	Changed                bool            `json:"changed"`
	SerialRank             []int           `json:"serial_candidate_rank"`
	PackedRank             []int           `json:"packed_candidate_rank"`
	RankChanged            bool            `json:"rank_changed"`
	SerialMargin           float64         `json:"serial_probability_margin"`
	PackedMargin           float64         `json:"packed_probability_margin"`
	Probability            errorStats      `json:"probability_error"`
	NormalisedPathLogScore errorStats      `json:"normalised_path_log_score_error"`
	ThresholdCrossings     []string        `json:"candidate_threshold_crossings,omitempty"`
}
type comparison struct {
	Rows          int               `json:"packed_token_rows"`
	Raw           errorStats        `json:"raw_logit_error"`
	Centred       errorStats        `json:"centred_node_logit_error"`
	Fields        []fieldComparison `json:"fields"`
	ChangedFields int               `json:"changed_fields"`
}
type caseReport struct {
	RequestIndex int          `json:"cohort_request_index"`
	ContextStart int          `json:"cohort_context_start"`
	Request      gso.Request  `json:"request"`
	Trials       []trial      `json:"trials"`
	Comparisons  []comparison `json:"comparisons"`
}
type report struct {
	Description string       `json:"description"`
	Started     string       `json:"started_utc"`
	Source      string       `json:"source_revision"`
	CohortSHA   string       `json:"cohort_sha256"`
	ModelSHA    string       `json:"model_sha256"`
	Device      string       `json:"device"`
	Thresholds  []float64    `json:"diagnostic_thresholds"`
	Cases       []caseReport `json:"cases"`
}

// Capture only the raw scores already consumed by the engine. It does not run
// another forward pass or substitute the independent-prefill reference.
type recorder struct {
	*gso.Gemma4NVIDIAScorer
	logits [][][]float32
}

func (r *recorder) ScoreSplitContextTrees(ctx context.Context, shared []int, contexts [][]int, branches []gso.Branch, cache bool) ([][][]float32, error) {
	out, err := r.Gemma4NVIDIAScorer.ScoreSplitContextTrees(ctx, shared, contexts, branches, cache)
	r.logits = out
	return out, err
}
func margin(c []gso.CandidateResult) float64 {
	a, b := 0., 0.
	for _, x := range c {
		if x.Probability >= a {
			a, b = x.Probability, a
		} else if x.Probability > b {
			b = x.Probability
		}
	}
	return a - b
}

// Ranks are candidate indices, keeping equal probabilities in input order.
func rank(c []gso.CandidateResult) []int {
	out := make([]int, len(c))
	for i := range out {
		out[i] = i
	}
	sort.SliceStable(out, func(i, j int) bool { return c[out[i]].Probability > c[out[j]].Probability })
	return out
}
func compare(a, b trial, thresholds []float64) (comparison, error) {
	out := comparison{Rows: b.Rows}
	if len(a.Logits) != len(b.Logits) || len(a.Response.Results) != len(b.Response.Results) {
		return out, fmt.Errorf("context count differs")
	}
	for i, ctx := range a.Logits {
		if len(ctx) != len(b.Logits[i]) {
			return out, fmt.Errorf("node count differs")
		}
		for j, row := range ctx {
			other := b.Logits[i][j]
			if len(row) == 0 || len(row) != len(other) {
				return out, fmt.Errorf("candidate logit count differs")
			}
			meanA, meanB := 0., 0.
			for k, v := range row {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.IsNaN(float64(other[k])) || math.IsInf(float64(other[k]), 0) {
					return out, fmt.Errorf("nonfinite logits")
				}
				meanA += float64(v)
				meanB += float64(other[k])
			}
			meanA /= float64(len(row))
			meanB /= float64(len(row))
			for k, v := range row {
				out.Raw.add(float64(other[k]) - float64(v))
				out.Centred.add((float64(other[k]) - meanB) - (float64(v) - meanA))
			}
		}
	}
	for i, result := range a.Response.Results {
		if len(result.Fields) != len(b.Response.Results[i].Fields) {
			return out, fmt.Errorf("field count differs")
		}
		// Sorted schema keys keep machine-readable output deterministic.
		names := make([]string, 0, len(result.Fields))
		for name := range result.Fields {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			af := result.Fields[name]
			bf, ok := b.Response.Results[i].Fields[name]
			if !ok || len(af.Candidates) == 0 || len(af.Candidates) != len(bf.Candidates) {
				return out, fmt.Errorf("field candidates differ")
			}
			f := fieldComparison{Context: i, Field: name, SerialWinner: af.Value, PackedWinner: bf.Value, Changed: string(af.Value) != string(bf.Value), SerialMargin: margin(af.Candidates), PackedMargin: margin(bf.Candidates)}
			f.SerialRank, f.PackedRank = rank(af.Candidates), rank(bf.Candidates)
			for j, v := range f.SerialRank {
				if v != f.PackedRank[j] {
					f.RankChanged = true
				}
			}
			if f.Changed {
				out.ChangedFields++
			}
			for j, x := range af.Candidates {
				y := bf.Candidates[j]
				if string(x.Value) != string(y.Value) {
					return out, fmt.Errorf("candidate order differs")
				}
				if math.IsNaN(x.Probability) || math.IsNaN(y.Probability) || x.Probability < 0 || y.Probability < 0 || x.Probability > 1 || y.Probability > 1 {
					return out, fmt.Errorf("invalid probability")
				}
				f.Probability.add(y.Probability - x.Probability)
				// log(p) is the complete path log score after field normalisation.
				if x.Probability > 0 && y.Probability > 0 {
					f.NormalisedPathLogScore.add(math.Log(y.Probability) - math.Log(x.Probability))
				}
				for _, threshold := range thresholds {
					if (x.Probability >= threshold) != (y.Probability >= threshold) {
						f.ThresholdCrossings = append(f.ThresholdCrossings, fmt.Sprintf("%s @ %g", x.Value, threshold))
					}
				}
			}
			out.Fields = append(out.Fields, f)
		}
	}
	return out, nil
}
func device(ctx context.Context) (string, int, error) {
	cctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	b, e := exec.CommandContext(cctx, "nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,memory.used,clocks.sm", "--format=csv,noheader,nounits", "-i", "0").Output()
	if e != nil {
		return "", 0, e
	}
	text := strings.TrimSpace(string(b))
	parts := strings.Split(text, ",")
	if len(parts) != 5 {
		return text, 0, fmt.Errorf("invalid GPU query")
	}
	temp, e := strconv.Atoi(strings.TrimSpace(parts[2]))
	return text, temp, e
}
func cool(ctx context.Context) error {
	for i := 0; i < 180; i++ {
		_, temp, e := device(ctx)
		if e != nil {
			return e
		}
		if temp <= 55 {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Second):
		}
	}
	return fmt.Errorf("GPU did not cool to 55C")
}
func decide(ctx context.Context, e *gso.Engine, request gso.Request) (gso.Response, error) {
	run, cancel := context.WithCancel(ctx)
	defer cancel()
	done := make(chan struct{})
	failure := make(chan error, 1)
	go func() {
		defer close(done)
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-run.Done():
				return
			case <-ticker.C:
				_, temp, err := device(run)
				if run.Err() != nil {
					return
				}
				if err != nil || temp >= 83 {
					if err == nil {
						err = fmt.Errorf("GPU thermal ceiling 83C reached")
					}
					failure <- err
					cancel()
					return
				}
			}
		}
	}()
	result, err := e.Decide(run, request)
	cancel()
	<-done
	select {
	case safety := <-failure:
		return gso.Response{}, safety
	default:
	}
	return result, err
}
func save(path string, r report) error {
	b, e := json.MarshalIndent(r, "", "  ")
	if e != nil {
		return e
	}
	if e = os.MkdirAll(filepath.Dir(path), 0755); e != nil {
		return e
	}
	if e = os.WriteFile(path+".tmp", append(b, '\n'), 0644); e != nil {
		return e
	}
	return os.Rename(path+".tmp", path)
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	modelPath := flag.String("model", "", "pinned GGUF path")
	tokPath := flag.String("tokenizer-dir", "", "pinned tokenizer directory")
	input := flag.String("cohort", "docs/benchmarks/multifield-cohort.json", "frozen requests")
	output := flag.String("out", "dist/benchmarks/multifield-precision.json", "report path")
	source := flag.String("source-revision", "", "source SHA (include dirty qualifier when applicable)")
	chunk := flag.Int("contexts-per-call", 4, "split frozen requests into cooled calls; does not change context text or order")
	resume := flag.Bool("resume", false, "resume complete chunks from the report after verifying source, cohort and request identities")
	analyse := flag.String("analyse", "", "recompute comparisons from a saved report without model execution")
	flag.Parse()
	if *chunk < 1 || *chunk > gso.MaxContexts {
		return fmt.Errorf("contexts-per-call must be 1..%d", gso.MaxContexts)
	}
	if *analyse != "" {
		b, err := os.ReadFile(*analyse)
		if err != nil {
			return err
		}
		var r report
		if err = json.Unmarshal(b, &r); err != nil {
			return err
		}
		if len(r.Cases) == 0 {
			return fmt.Errorf("report has no cases")
		}
		for i := range r.Cases {
			c := &r.Cases[i]
			if len(c.Trials) < 2 || c.Trials[0].Rows != 0 {
				return fmt.Errorf("case %d has no serial reference", i)
			}
			c.Comparisons = nil
			for _, tr := range c.Trials[1:] {
				cmp, err := compare(c.Trials[0], tr, r.Thresholds)
				if err != nil {
					return err
				}
				c.Comparisons = append(c.Comparisons, cmp)
			}
		}
		return save(*output, r)
	}
	if *modelPath == "" || *tokPath == "" || *source == "" {
		return fmt.Errorf("-model, -tokenizer-dir and -source-revision required")
	}
	b, err := os.ReadFile(*input)
	if err != nil {
		return err
	}
	var c cohort
	if err = json.Unmarshal(b, &c); err != nil {
		return err
	}
	if len(c.Requests) == 0 {
		return fmt.Errorf("empty cohort")
	}
	for _, v := range c.Thresholds {
		if v <= 0 || v >= 1 {
			return fmt.Errorf("invalid threshold")
		}
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if procs, e := exec.CommandContext(ctx, "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits").Output(); e != nil {
		return e
	} else if strings.TrimSpace(string(procs)) != "" {
		return fmt.Errorf("GPU has active compute processes")
	}
	if err = gso.VerifyV1Artifacts(*modelPath, *tokPath); err != nil {
		return err
	}
	tok, err := tokenizer.LoadWithConfig(*tokPath)
	if err != nil {
		return err
	}
	m, err := model.LoadGemma4GGUFAsLlama(*modelPath)
	if err != nil {
		return err
	}
	m.Tok = tok
	gpu, err := model.NewGemma4NVIDIA(m)
	if err != nil {
		return err
	}
	defer gpu.Close()
	s := &recorder{Gemma4NVIDIAScorer: &gso.Gemma4NVIDIAScorer{Model: m, GPU: gpu}}
	defer s.Close()
	e := &gso.Engine{Tokenizer: tok, Scorer: s, BOSToken: m.Config.BOSTokenID}
	hash := sha256.Sum256(b)
	dev, _, err := device(ctx)
	if err != nil {
		return err
	}
	r := report{Description: c.Description + " Timings include instrumentation and are descriptive, not a latency distribution. Serial depth batches may use F32 activations; packed projection rows use Q8. Thresholds are diagnostic, not deployment policy.", Started: time.Now().UTC().Format(time.RFC3339), Source: *source, CohortSHA: hex.EncodeToString(hash[:]), ModelSHA: gso.V1Provenance.ModelSHA256, Device: dev, Thresholds: c.Thresholds}
	if *resume {
		if err := resumeReport(*output, &r, c, *chunk); err != nil {
			return err
		}
	}
	completed := len(r.Cases)
	ordinal := 0
	for i, request := range c.Requests {
		if err = request.NormalizeAndValidate(); err != nil {
			return err
		}
		if request.Mode != gso.ModeTree {
			return fmt.Errorf("comparison requires tree mode")
		}
		for start := 0; start < len(request.Contexts); start += *chunk {
			ordinal++
			if ordinal <= completed {
				continue
			}
			part := request
			part.Contexts = request.Contexts[start:min(start+*chunk, len(request.Contexts))]
			cr := caseReport{RequestIndex: i, ContextStart: start, Request: part}
			for _, rows := range []int{0, 128, 256, 512} {
				s.PackedTokenRows = rows
				if err = cool(ctx); err != nil {
					return err
				}
				warmup := part
				warmup.Contexts = part.Contexts[:1]
				if _, err = decide(ctx, e, warmup); err != nil {
					return err
				}
				if err = cool(ctx); err != nil {
					return err
				}
				before, _, err := device(ctx)
				if err != nil {
					return err
				}
				s.logits = nil
				response, err := decide(ctx, e, part)
				if err != nil {
					return err
				}
				after, _, err := device(ctx)
				if err != nil {
					return err
				}
				if len(s.logits) != len(part.Contexts) {
					return fmt.Errorf("scorer did not return raw scores")
				}
				tr := trial{Rows: rows, Response: response, Logits: s.logits, DeviceBefore: before, DeviceAfter: after}
				cr.Trials = append(cr.Trials, tr)
				if rows > 0 {
					cmp, err := compare(cr.Trials[0], tr, c.Thresholds)
					if err != nil {
						return err
					}
					cr.Comparisons = append(cr.Comparisons, cmp)
					fmt.Printf("case=%d start=%d rows=%d fields=%d changed=%d raw_max=%g centred_max=%g\n", i, start, rows, len(cmp.Fields), cmp.ChangedFields, cmp.Raw.Max, cmp.Centred.Max)
				}
			}
			r.Cases = append(r.Cases, cr)
			if err = save(*output, r); err != nil {
				return err
			}
		}
	}
	fmt.Println("report:", *output)
	return nil
}

func resumeReport(path string, target *report, c cohort, chunk int) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var old report
	if err = json.Unmarshal(b, &old); err != nil {
		return err
	}
	if old.CohortSHA != target.CohortSHA || old.ModelSHA != target.ModelSHA || old.Source != target.Source {
		return fmt.Errorf("resume provenance mismatch")
	}
	index := 0
	for i, request := range c.Requests {
		if err = request.NormalizeAndValidate(); err != nil {
			return err
		}
		for start := 0; start < len(request.Contexts); start += chunk {
			if index >= len(old.Cases) {
				break
			}
			part := request
			part.Contexts = request.Contexts[start:min(start+chunk, len(request.Contexts))]
			got := old.Cases[index]
			wantBytes, _ := json.Marshal(part)
			gotBytes, _ := json.Marshal(got.Request)
			if got.RequestIndex != i || got.ContextStart != start || string(wantBytes) != string(gotBytes) || len(got.Trials) != 4 || len(got.Comparisons) != 3 {
				return fmt.Errorf("resume chunk %d mismatch or incomplete", index)
			}
			for j, rows := range []int{0, 128, 256, 512} {
				if got.Trials[j].Rows != rows {
					return fmt.Errorf("resume row budget mismatch")
				}
			}
			index++
		}
	}
	if index != len(old.Cases) {
		return fmt.Errorf("resume has extra cases")
	}
	*target = old
	return nil
}
