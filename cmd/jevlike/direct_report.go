package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"time"

	"github.com/rcarmo/go-pherence/model/jevlike"
)

type directMatchedControl struct {
	Task            string  `json:"task"`
	Variant         string  `json:"variant"`
	Examples        int     `json:"examples"`
	NormalAccuracy  float64 `json:"normal_accuracy"`
	ControlAccuracy float64 `json:"control_accuracy"`
	Changed         int     `json:"changed"`
}

type directMetricGroup struct {
	Task       string                   `json:"task"`
	Variant    string                   `json:"variant"`
	Rejected   int                      `json:"rejected"`
	Metrics    *jevlike.DecisionMetrics `json:"metrics,omitempty"`
	P50Seconds float64                  `json:"p50_seconds"`
	P95Seconds float64                  `json:"p95_seconds"`
}

func runDirectReport(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("direct-report", flag.ContinueOnError)
	fs.SetOutput(stderr)
	path := fs.String("results", "", "score-batch output JSONL")
	spec := fs.String("selection", "", "Request selection manifest carrying partition and expected row count")
	requests := fs.String("requests", "", "Exact hash-verified request JSONL")
	dataset := fs.String("dataset", "", "Prepared dataset with manifest and provenance")
	fit := fs.Bool("fit-temperature", false, "Fit only when selection explicitly declares calibration partition")
	calibration := fs.String("calibration", "", "Frozen calibration report; exact model/dataset identity required")
	if err := parseSubcommandFlags(fs, args); err != nil {
		if err == errHelpRequested {
			return nil
		}
		return err
	}
	if *path == "" || *spec == "" || *requests == "" || *dataset == "" {
		return fmt.Errorf("results, selection, requests and dataset required")
	}
	selection, expected, err := readDirectSelection(*spec, *requests, *dataset)
	if err != nil {
		return err
	}
	if *fit && *calibration != "" {
		return fmt.Errorf("cannot fit and apply temperature together")
	}
	if *fit && selection.Partition != "calibration" {
		return fmt.Errorf("temperature fitting requires calibration partition; never fit validation/test")
	}
	f, err := os.Open(*path)
	if err != nil {
		return err
	}
	defer f.Close()
	type group struct {
		logits        [][]float32
		labels        []int
		latencies     []float64
		rejected      int
		task, variant string
	}
	groups := map[string]*group{}
	seen := map[string]bool{}
	models := map[string]bool{}
	rows := 0
	predictions := map[string]map[string]string{}
	goldByExample := map[string]string{}
	taskByExample := map[string]string{}
	scan := bufio.NewScanner(f)
	scan.Buffer(make([]byte, 64*1024), 4<<20)
	var calLogits [][]float32
	var calLabels []int
	for scan.Scan() {
		var row directBatchOutput
		if err = json.Unmarshal(scan.Bytes(), &row); err != nil {
			return err
		}
		rows++
		key := row.Task + "\x00" + row.ID + "\x00" + row.Variant
		if seen[key] {
			return fmt.Errorf("duplicate result row %s", row.ID)
		}
		seen[key] = true
		request, ok := expected[key]
		if !ok || request.GoldID != row.GoldID || row.ModelID == "" || math.IsNaN(row.Elapsed) || math.IsInf(row.Elapsed, 0) || row.Elapsed < 0 {
			return fmt.Errorf("result not bound to request or invalid metadata: %s", row.ID)
		}
		if (row.Error != "") == (row.Result != nil) {
			return fmt.Errorf("exactly one result or rejection required")
		}
		if row.Result != nil {
			r := row.Result
			if len(r.IDs) != len(request.Request.Candidates) || len(r.Logits) != len(r.IDs) {
				return fmt.Errorf("result candidate shape mismatch")
			}
			best := 0
			for i, c := range request.Request.Candidates {
				if c.ID != r.IDs[i] || math.IsNaN(float64(r.Logits[i])) || math.IsInf(float64(r.Logits[i]), 0) {
					return fmt.Errorf("result candidate/logit mismatch")
				}
				if r.Logits[i] > r.Logits[best] {
					best = i
				}
			}
			if r.SelectedID != r.IDs[best] {
				return fmt.Errorf("reported selection disagrees with logits")
			}
		}
		models[row.ModelID] = true
		for _, task := range []string{row.Task, "all"} {
			k := task + "\x00" + row.Variant
			g := groups[k]
			if g == nil {
				g = &group{task: task, variant: row.Variant}
				groups[k] = g
			}
			if row.Error != "" {
				g.rejected++
				continue
			}
			if row.Result == nil || len(row.Result.IDs) != len(row.Result.Logits) {
				return fmt.Errorf("missing result")
			}
			label := -1
			for i, id := range row.Result.IDs {
				if id == row.GoldID {
					label = i
				}
			}
			if label < 0 {
				return fmt.Errorf("gold ID absent from result")
			}
			g.logits = append(g.logits, row.Result.Logits)
			g.labels = append(g.labels, label)
			g.latencies = append(g.latencies, row.Elapsed)
			if task == row.Task && row.Variant == "normal" {
				calLogits = append(calLogits, row.Result.Logits)
				calLabels = append(calLabels, label)
			}
		}
		if row.Result != nil {
			key := row.Task + "\x00" + row.ID
			if predictions[key] == nil {
				predictions[key] = map[string]string{}
			}
			predictions[key][row.Variant] = row.Result.SelectedID
			goldByExample[key] = row.GoldID
			taskByExample[key] = row.Task
		}
	}
	if err = scan.Err(); err != nil {
		return err
	}
	if rows != selection.Requests {
		return fmt.Errorf("incomplete results: %d of %d", rows, selection.Requests)
	}
	if len(models) != 1 {
		return fmt.Errorf("mixed checkpoint identities")
	}
	temperature := float64(1)
	modelID := ""
	for id := range models {
		modelID = id
	}
	calibrationSHA256 := ""
	if *calibration != "" {
		b, err := os.ReadFile(*calibration)
		if err != nil {
			return err
		}
		var cal struct {
			Version              int     `json:"version"`
			Partition            string  `json:"partition"`
			Fitted               bool    `json:"temperature_fitted"`
			Temperature          float64 `json:"temperature"`
			ModelID              string  `json:"model_id"`
			SourceManifestSHA256 string  `json:"source_manifest_sha256"`
			RequestSHA256        string  `json:"request_sha256"`
			ResultSHA256         string  `json:"result_sha256"`
		}
		if err = json.Unmarshal(b, &cal); err != nil {
			return err
		}
		if cal.Version != 2 || cal.Partition != "calibration" || !cal.Fitted || cal.ModelID != modelID || cal.SourceManifestSHA256 != selection.SourceManifestSHA256 || cal.RequestSHA256 == selection.RequestSHA256 || len(cal.ResultSHA256) != 64 || len(cal.RequestSHA256) != 64 || math.IsNaN(cal.Temperature) || math.IsInf(cal.Temperature, 0) || cal.Temperature < 0.05 || cal.Temperature > 20 {
			return fmt.Errorf("incompatible frozen calibration report")
		}
		temperature = cal.Temperature
		calibrationSHA256 = directHash(b)
	}
	if *fit {
		temperature, err = jevlike.FitChoiceTemperature(calLogits, calLabels)
		if err != nil {
			return err
		}
	}
	keys := make([]string, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var report []directMetricGroup
	for _, k := range keys {
		g := groups[k]
		row := directMetricGroup{Task: g.task, Variant: g.variant, Rejected: g.rejected}
		if len(g.logits) > 0 {
			m, err := jevlike.DecisionMetricsAtTemperature(g.logits, g.labels, temperature)
			if err != nil {
				return err
			}
			row.Metrics = &m
			sort.Float64s(g.latencies)
			row.P50Seconds = g.latencies[int(math.Ceil(float64(len(g.latencies))*0.50))-1]
			row.P95Seconds = g.latencies[int(math.Ceil(float64(len(g.latencies))*0.95))-1]
		}
		report = append(report, row)
	}
	paired := map[string]*directMatchedControl{}
	for key, p := range predictions {
		normal, ok := p["normal"]
		if !ok {
			continue
		}
		for variant, control := range p {
			if variant == "normal" {
				continue
			}
			for _, task := range []string{taskByExample[key], "all"} {
				k := task + "\x00" + variant
				if paired[k] == nil {
					paired[k] = &directMatchedControl{Task: task, Variant: variant}
				}
				pair := paired[k]
				pair.Examples++
				if normal == goldByExample[key] {
					pair.NormalAccuracy++
				}
				if control == goldByExample[key] {
					pair.ControlAccuracy++
				}
				if normal != control {
					pair.Changed++
				}
			}
		}
	}
	var matched []directMatchedControl
	for _, pair := range paired {
		pair.NormalAccuracy /= float64(pair.Examples)
		pair.ControlAccuracy /= float64(pair.Examples)
		matched = append(matched, *pair)
	}
	sort.Slice(matched, func(i, j int) bool {
		if matched[i].Task != matched[j].Task {
			return matched[i].Task < matched[j].Task
		}
		return matched[i].Variant < matched[j].Variant
	})
	compared, changed := 0, 0
	for _, p := range predictions {
		a, ok := p["normal"]
		b, ok2 := p["reverse-options"]
		if ok && ok2 {
			compared++
			if a != b {
				changed++
			}
		}
	}
	resultBytes, err := os.ReadFile(*path)
	if err != nil {
		return err
	}
	return writeJSON(stdout, struct {
		CalibrationSHA256    string                 `json:"calibration_sha256,omitempty"`
		Matched              []directMatchedControl `json:"matched_controls"`
		ModelID              string                 `json:"model_id"`
		ResultSHA256         string                 `json:"result_sha256"`
		RequestSHA256        string                 `json:"request_sha256"`
		SourceManifestSHA256 string                 `json:"source_manifest_sha256"`
		Version              int                    `json:"version"`
		Partition            string                 `json:"partition"`
		Temperature          float64                `json:"temperature"`
		Fitted               bool                   `json:"temperature_fitted"`
		Groups               []directMetricGroup    `json:"groups"`
		OrderCompared        int                    `json:"order_compared"`
		OrderChanged         int                    `json:"order_changed"`
		Generated            string                 `json:"generated_utc"`
	}{calibrationSHA256, matched, modelID, directHash(resultBytes), selection.RequestSHA256, selection.SourceManifestSHA256, 2, selection.Partition, temperature, *fit, report, compared, changed, time.Now().UTC().Format(time.RFC3339)})
}
