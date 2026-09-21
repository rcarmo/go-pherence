package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"

	"github.com/rcarmo/go-pherence/model/jevlike"
)

type scoreReportGroup struct {
	Group                           string                   `json:"group"`
	Variant                         string                   `json:"variant"`
	Requested                       int                      `json:"requested"`
	Admitted                        int                      `json:"admitted"`
	Rejected                        int                      `json:"rejected"`
	AccuracyCountingRejectionsWrong float64                  `json:"accuracy_rejections_wrong"`
	RandomAccuracyAllRequested      float64                  `json:"random_accuracy_all_requested"`
	MetricsAdmitted                 *jevlike.DecisionMetrics `json:"metrics_admitted,omitempty"`
	DecisionSecondsTotal            float64                  `json:"decision_seconds_total"`
	RejectionCauses                 map[string]int           `json:"rejection_causes"`
}

type scoreReport struct {
	Version           int                `json:"version"`
	Scope             string             `json:"scope"`
	Cohort            string             `json:"cohort"`
	Arm               string             `json:"arm"`
	ModelID           string             `json:"model_id"`
	SelectionSHA256   string             `json:"selection_sha256"`
	RequestsSHA256    string             `json:"requests_sha256"`
	ResultsSHA256     string             `json:"results_sha256"`
	ScoreSemantics    string             `json:"score_semantics"`
	TemperatureFitted bool               `json:"temperature_fitted"`
	Groups            []scoreReportGroup `json:"groups"`
}

func readOutputs(payload []byte, inputs []input) ([]output, string, string, error) {
	scan := bufio.NewScanner(bytes.NewReader(payload))
	scan.Buffer(make([]byte, 64<<10), 2<<20)
	rows := make([]output, 0, len(inputs))
	arm, modelID := "", ""
	for scan.Scan() {
		if len(rows) >= len(inputs) {
			return nil, "", "", errors.New("results contain excess rows")
		}
		var row output
		decoder := json.NewDecoder(bytes.NewReader(scan.Bytes()))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&row); err != nil {
			return nil, "", "", fmt.Errorf("result row %d: %w", len(rows)+1, err)
		}
		if decoder.Decode(new(any)) != io.EOF {
			return nil, "", "", fmt.Errorf("result row %d has trailing data", len(rows)+1)
		}
		if arm == "" {
			arm, modelID = row.Arm, row.ModelID
		}
		if arm == "" || modelID == "" {
			return nil, "", "", errors.New("result arm/model identity missing")
		}
		if err := validateOutput(row, inputs[len(rows)], arm, modelID); err != nil {
			return nil, "", "", fmt.Errorf("result row %d: %w", len(rows)+1, err)
		}
		rows = append(rows, row)
	}
	if err := scan.Err(); err != nil {
		return nil, "", "", err
	}
	if len(rows) != len(inputs) {
		return nil, "", "", fmt.Errorf("result row count %d want %d", len(rows), len(inputs))
	}
	return rows, arm, modelID, nil
}

func buildScoreGroup(name, variant string, indices []int, inputs []input, outputs []output) (scoreReportGroup, error) {
	group := scoreReportGroup{Group: name, Variant: variant, Requested: len(indices), RejectionCauses: map[string]int{}}
	if len(indices) == 0 {
		return group, errors.New("empty score group")
	}
	var probabilities [][]float32
	var labels []int
	correct := 0
	for _, index := range indices {
		in, out := inputs[index], outputs[index]
		group.DecisionSecondsTotal += out.DecisionSeconds
		group.RandomAccuracyAllRequested += 1 / float64(len(in.Request.Candidates))
		if out.Error != "" {
			group.Rejected++
			group.RejectionCauses[out.Error]++
			continue
		}
		label := choiceIndex(in, in.GoldID)
		if label < 0 || len(out.Options) != len(in.Request.Candidates) {
			return group, errors.New("invalid admitted result shape")
		}
		row := make([]float32, len(out.Options))
		for i, option := range out.Options {
			row[i] = option.Score
		}
		probabilities = append(probabilities, row)
		labels = append(labels, label)
		group.Admitted++
		if out.SelectedID == in.GoldID {
			correct++
		}
	}
	group.RandomAccuracyAllRequested /= float64(group.Requested)
	group.AccuracyCountingRejectionsWrong = float64(correct) / float64(group.Requested)
	if group.Admitted > 0 {
		metrics, err := jevlike.DecisionMetricsFromProbabilities(probabilities, labels)
		if err != nil {
			return group, err
		}
		group.MetricsAdmitted = &metrics
	}
	return group, nil
}

func runScoreReport(study, cohort, results, destination string) error {
	if study == "" || results == "" || destination == "" || (cohort != "screening" && cohort != "finalist") {
		return errors.New("study/cohort/report-results/report-output required")
	}
	selectionBytes, err := os.ReadFile(filepath.Join(study, "selection.json"))
	if err != nil {
		return err
	}
	var selected selection
	if err = json.Unmarshal(selectionBytes, &selected); err != nil {
		return err
	}
	if selected.Scope != "jev-port-bakeoff-v1" {
		return errors.New("wrong study scope")
	}
	cohortSpec, ok := selected.Cohorts[cohort]
	if !ok {
		return errors.New("cohort absent")
	}
	requestBytes, err := os.ReadFile(filepath.Join(study, cohortSpec.Path))
	if err != nil {
		return err
	}
	if digest(requestBytes) != cohortSpec.SHA256 {
		return errors.New("cohort hash mismatch")
	}
	inputs, err := readInputs(requestBytes)
	if err != nil {
		return err
	}
	if len(inputs) != cohortSpec.Rows {
		return fmt.Errorf("cohort row count %d want %d", len(inputs), cohortSpec.Rows)
	}
	resultBytes, err := os.ReadFile(results)
	if err != nil {
		return err
	}
	outputs, arm, modelID, err := readOutputs(resultBytes, inputs)
	if err != nil {
		return err
	}
	variants := map[string][]int{}
	tasks := map[string][]int{}
	for i, in := range inputs {
		variants[in.Variant] = append(variants[in.Variant], i)
		if in.Variant == "normal" {
			tasks[in.Task] = append(tasks[in.Task], i)
		}
	}
	var groups []scoreReportGroup
	variantNames := make([]string, 0, len(variants))
	for variant := range variants {
		variantNames = append(variantNames, variant)
	}
	sort.Strings(variantNames)
	for _, variant := range variantNames {
		group, err := buildScoreGroup("all", variant, variants[variant], inputs, outputs)
		if err != nil {
			return err
		}
		groups = append(groups, group)
	}
	taskNames := make([]string, 0, len(tasks))
	for task := range tasks {
		taskNames = append(taskNames, task)
	}
	sort.Strings(taskNames)
	for _, task := range taskNames {
		group, err := buildScoreGroup(task, "normal", tasks[task], inputs, outputs)
		if err != nil {
			return err
		}
		groups = append(groups, group)
	}
	report := scoreReport{
		Version: 1, Scope: selected.Scope, Cohort: cohort, Arm: arm, ModelID: modelID,
		SelectionSHA256: digest(selectionBytes), RequestsSHA256: digest(requestBytes), ResultsSHA256: digest(resultBytes),
		ScoreSemantics: "stored candidate scores normalized within each offered set; no held-out calibration fit", TemperatureFitted: false, Groups: groups,
	}
	payload, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	file, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	if _, err = file.Write(append(payload, '\n')); err != nil {
		file.Close()
		return err
	}
	if err = file.Sync(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}
