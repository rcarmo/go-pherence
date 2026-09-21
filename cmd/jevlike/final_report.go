package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/rcarmo/go-pherence/model/jevlike"
)

type finalMetricGroup struct {
	Group                           string                   `json:"group"`
	Requested                       int                      `json:"requested"`
	Admitted                        int                      `json:"admitted"`
	Rejected                        int                      `json:"rejected"`
	AccuracyCountingRejectionsWrong float64                  `json:"accuracy_rejections_wrong"`
	RandomAccuracyAllRequested      float64                  `json:"random_accuracy_all_requested"`
	Raw                             *jevlike.DecisionMetrics `json:"raw_admitted,omitempty"`
	Calibrated                      *jevlike.DecisionMetrics `json:"calibrated_admitted,omitempty"`
	RejectionCauses                 map[string]int           `json:"rejection_causes"`
	DecisionSeconds                 float64                  `json:"decision_seconds_total"`
}

func finalGroupMetrics(name string, rows []directBatchInput, outcomes []directBatchOutput, temp float64) (finalMetricGroup, error) {
	g := finalMetricGroup{Group: name, Requested: len(rows), RejectionCauses: map[string]int{}}
	if len(rows) != len(outcomes) || len(rows) == 0 {
		return g, fmt.Errorf("empty or mismatched final group")
	}
	var logits [][]float32
	var labels []int
	correct := 0
	for i, row := range rows {
		g.RandomAccuracyAllRequested += 1 / float64(len(row.Request.Candidates))
		r := outcomes[i]
		g.DecisionSeconds += r.Elapsed
		if r.Error != "" {
			g.Rejected++
			g.RejectionCauses[r.Error]++
			continue
		}
		if r.Result == nil {
			return g, fmt.Errorf("missing final outcome")
		}
		label := -1
		for j, c := range row.Request.Candidates {
			if c.ID == row.GoldID {
				label = j
			}
		}
		if label < 0 {
			return g, fmt.Errorf("missing gold")
		}
		logits = append(logits, r.Result.Logits)
		labels = append(labels, label)
		g.Admitted++
		if r.Result.SelectedID == row.GoldID {
			correct++
		}
	}
	g.RandomAccuracyAllRequested /= float64(g.Requested)
	g.AccuracyCountingRejectionsWrong = float64(correct) / float64(g.Requested)
	if len(logits) > 0 {
		m, e := jevlike.DecisionMetricsAtTemperature(logits, labels, 1)
		if e != nil {
			return g, e
		}
		c, e := jevlike.DecisionMetricsAtTemperature(logits, labels, temp)
		if e != nil {
			return g, e
		}
		g.Raw = &m
		g.Calibrated = &c
	}
	return g, nil
}
func runFinalReport(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("final-report", flag.ContinueOnError)
	fs.SetOutput(stderr)
	freezePath := fs.String("freeze", "", "Committed pre-test freeze")
	study := fs.String("study", "", "Final request directory")
	records := fs.String("records", "", "Immutable original records")
	output := fs.String("output", "", "New report directory")
	if e := parseSubcommandFlags(fs, args); e != nil {
		if e == errHelpRequested {
			return nil
		}
		return e
	}
	if *freezePath == "" || *study == "" || *records == "" || *output == "" {
		return fmt.Errorf("freeze/study/records/output required")
	}
	f, freezeHash, e := verifyFinalFreeze(*freezePath)
	if e != nil {
		return e
	}
	selectionBytes, e := os.ReadFile(filepath.Join(*study, "selection.json"))
	if e != nil {
		return e
	}
	var gate struct {
		Freeze string `json:"final_freeze_sha256"`
	}
	if e = json.Unmarshal(selectionBytes, &gate); e != nil {
		return e
	}
	if gate.Freeze != freezeHash {
		return fmt.Errorf("wrong final freeze")
	}
	s, expected, e := readDirectSelection(filepath.Join(*study, "selection.json"), filepath.Join(*study, "requests.jsonl"), f.Dataset)
	if e != nil {
		return e
	}
	if s.Partition != "test" || s.Requests != f.Expected || s.PartitionSHA256 != f.PartitionSHA256 || s.SourceManifestSHA256 != f.DatasetSHA256 {
		return fmt.Errorf("not complete frozen test selection")
	}
	rows, e := orderedFeatureRequests(filepath.Join(*study, "requests.jsonl"), expected)
	if e != nil {
		return e
	}
	contract, e := jevlike.LoadFeatureContract(filepath.Dir(f.Contract))
	if e != nil {
		return e
	}
	models := map[string]string{"instruction": f.ModelID}
	calPaths := map[string]string{"instruction": f.DirectCalibration}
	arms := []string{"instruction"}
	for _, h := range f.Heads {
		arm := fmt.Sprintf("seed-%d", h.Seed)
		arms = append(arms, arm)
		models[arm] = "jevlike-head@" + f.Files[h.Checkpoint] + "/feature:" + contract.ID()
		calPaths[arm] = h.Calibration
	}
	temperatures := map[string]float64{}
	for arm, path := range calPaths {
		b, e := os.ReadFile(path)
		if e != nil {
			return e
		}
		var c struct {
			Version     int
			Partition   string
			Fitted      bool `json:"temperature_fitted"`
			Temperature float64
			ModelID     string `json:"model_id"`
			Dataset     string `json:"source_manifest_sha256"`
		}
		if e = json.Unmarshal(b, &c); e != nil {
			return e
		}
		if c.Version != 2 || c.Partition != "calibration" || !c.Fitted || c.ModelID != models[arm] || c.Dataset != f.DatasetSHA256 || c.Temperature < .05 || c.Temperature > 20 {
			return fmt.Errorf("incompatible frozen calibration %s", arm)
		}
		temperatures[arm] = c.Temperature
	}
	dirs, e := os.ReadDir(*records)
	if e != nil {
		return e
	}
	jsonCount := 0
	for _, d := range dirs {
		if d.Name() == ".run.lock" {
			return fmt.Errorf("final run still owns records")
		}
		if strings.HasSuffix(d.Name(), ".json") {
			jsonCount++
		}
	}
	if jsonCount != len(rows) || len(rows) != f.Expected {
		return fmt.Errorf("incomplete final outcomes: %d of %d", jsonCount, f.Expected)
	}
	outcomes := map[string][]directBatchOutput{}
	common := make([]bool, len(rows))
	var artifacts []map[string]any
	var featureSeconds float64
	for i, row := range rows {
		key := directRowKey(row.Task, row.ID, row.Variant)
		path := filepath.Join(*records, fmt.Sprintf("%04d-%s.json", i+1, directHash([]byte(key))[:16]))
		b, e := os.ReadFile(path)
		if e != nil {
			return e
		}
		var entry finalEntry
		if e = json.Unmarshal(b, &entry); e != nil {
			return e
		}
		if e = validateFinalEntry(entry, freezeHash, s.RequestSHA256, row, models); e != nil {
			return e
		}
		common[i] = true
		featureSeconds += entry.FeatureSeconds
		for _, arm := range arms {
			outcomes[arm] = append(outcomes[arm], entry.Outcomes[arm])
			if entry.Outcomes[arm].Error != "" {
				common[i] = false
			}
		}
		artifacts = append(artifacts, map[string]any{"path": filepath.Base(path), "sha256": directHash(b)})
	}
	commonCount := 0
	for _, b := range common {
		if b {
			commonCount++
		}
	}
	groupIndices := map[string][]int{}
	for i, row := range rows {
		for _, g := range []string{"all", row.Task, fmt.Sprintf("choice-count/%d", len(row.Request.Candidates))} {
			groupIndices[g] = append(groupIndices[g], i)
		}
	}
	keys := make([]string, 0, len(groupIndices))
	for k := range groupIndices {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	reports := map[string]any{}
	meanAccuracy := float64(0)
	for _, arm := range arms {
		var groups, matched []finalMetricGroup
		for _, key := range keys {
			var r, cr []directBatchInput
			var o, co []directBatchOutput
			for _, i := range groupIndices[key] {
				r = append(r, rows[i])
				o = append(o, outcomes[arm][i])
				if common[i] {
					cr = append(cr, rows[i])
					co = append(co, outcomes[arm][i])
				}
			}
			g, e := finalGroupMetrics(key, r, o, temperatures[arm])
			if e != nil {
				return e
			}
			groups = append(groups, g)
			if key == "all" && arm != "instruction" {
				meanAccuracy += g.AccuracyCountingRejectionsWrong / 3
			}
			if len(cr) > 0 {
				c, e := finalGroupMetrics(key, cr, co, temperatures[arm])
				if e != nil {
					return e
				}
				matched = append(matched, c)
			}
		}
		reports[arm] = map[string]any{"model_id": models[arm], "temperature": temperatures[arm], "calibration_sha256": f.Files[calPaths[arm]], "groups": groups, "common_admitted_groups": matched}
	}
	if _, e = os.Stat(*output); !os.IsNotExist(e) {
		return fmt.Errorf("report output exists or unavailable")
	}
	if e = os.MkdirAll(*output, 0o755); e != nil {
		return e
	}
	for _, arm := range arms {
		file, e := os.OpenFile(filepath.Join(*output, arm+"-results.jsonl"), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
		if e != nil {
			return e
		}
		enc := json.NewEncoder(file)
		for _, r := range outcomes[arm] {
			if e = enc.Encode(r); e != nil {
				file.Close()
				return e
			}
		}
		if e = file.Sync(); e != nil {
			file.Close()
			return e
		}
		if e = file.Close(); e != nil {
			return e
		}
	}
	report := map[string]any{"version": 1, "partition": "test", "freeze_sha256": freezeHash, "request_sha256": s.RequestSHA256, "dataset_sha256": f.DatasetSHA256, "partition_sha256": f.PartitionSHA256, "requested": len(rows), "common_admitted": commonCount, "arms": reports, "head_mean_accuracy_rejections_wrong": meanAccuracy, "feature_extraction_seconds_total": featureSeconds, "record_artifacts": artifacts, "policy": "one frozen normal pass; no refit/tuning; calibration from existing separate split; historical validation failures retained; broader human-reviewed/transfer tests explicitly deferred; experiment completion not production promotion"}
	bytes, e := json.MarshalIndent(report, "", "  ")
	if e != nil {
		return e
	}
	if e = os.WriteFile(filepath.Join(*output, "report.json"), append(bytes, '\n'), 0o644); e != nil {
		return e
	}
	return writeJSON(stdout, map[string]any{"complete": true, "report": filepath.Join(*output, "report.json"), "requested": len(rows), "common_admitted": commonCount, "head_mean_accuracy": meanAccuracy})
}
