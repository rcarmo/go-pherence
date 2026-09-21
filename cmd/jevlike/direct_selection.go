package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type directSelection struct {
	Partition            string `json:"partition"`
	Requests             int    `json:"requests"`
	RequestSHA256        string `json:"request_sha256"`
	SourceManifestSHA256 string `json:"source_manifest_sha256"`
	PartitionSHA256      string `json:"partition_sha256"`
	ValidationSHA256     string `json:"validation_sha256"` // version-one pilot compatibility
}

func directHash(b []byte) string                   { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func directRowKey(task, id, variant string) string { return task + "\x00" + id + "\x00" + variant }

// Bind a study to verified requests AND the prepared partition/provenance.
// Relabelling a validation selection as calibration must not authorise a fit.
func readDirectSelection(spec, requests, dataset string) (directSelection, map[string]directBatchInput, error) {
	var s directSelection
	b, err := os.ReadFile(spec)
	if err != nil {
		return s, nil, err
	}
	if err = json.Unmarshal(b, &s); err != nil {
		return s, nil, err
	}
	if s.Partition != "train" && s.Partition != "validation" && s.Partition != "calibration" && s.Partition != "test" {
		return s, nil, fmt.Errorf("explicit evaluation partition required")
	}
	manifest, err := os.ReadFile(filepath.Join(dataset, "manifest.json"))
	if err != nil {
		return s, nil, err
	}
	if directHash(manifest) != s.SourceManifestSHA256 {
		return s, nil, fmt.Errorf("dataset manifest identity mismatch")
	}
	var m struct {
		Artifacts []struct{ Path, SHA256 string }
		Inputs    []struct{ Source, Config string }
	}
	if err = json.Unmarshal(manifest, &m); err != nil {
		return s, nil, err
	}
	hashes := map[string]string{}
	for _, a := range m.Artifacts {
		hashes[a.Path] = a.SHA256
	}
	verify := func(name string) ([]byte, error) {
		b, e := os.ReadFile(filepath.Join(dataset, name))
		if e != nil {
			return nil, e
		}
		if directHash(b) != hashes[name] {
			return nil, fmt.Errorf("dataset artifact identity mismatch: %s", name)
		}
		return b, nil
	}
	partition, err := verify(s.Partition + ".jsonl")
	if err != nil {
		return s, nil, err
	}
	expected := s.PartitionSHA256
	if expected == "" && s.Partition == "validation" {
		expected = s.ValidationSHA256
	}
	if expected != directHash(partition) {
		return s, nil, fmt.Errorf("partition identity mismatch")
	}
	prov, err := verify("provenance.jsonl")
	if err != nil {
		return s, nil, err
	}
	partitionRows := strings.Split(strings.TrimSpace(string(partition)), "\n")
	golds := map[string]string{}
	options := map[string]map[string]string{}
	for _, line := range strings.Split(strings.TrimSpace(string(prov)), "\n") {
		var p struct {
			Partition string
			SourceID  string `json:"source_id"`
			OutputRow int    `json:"output_row"`
			Input     int
		}
		if err = json.Unmarshal([]byte(line), &p); err != nil {
			return s, nil, err
		}
		if p.Partition != s.Partition {
			continue
		}
		if p.OutputRow < 1 || p.OutputRow > len(partitionRows) || p.Input < 0 || p.Input >= len(m.Inputs) {
			return s, nil, fmt.Errorf("invalid provenance row")
		}
		var ex struct {
			Options []string
			Label   int
		}
		if err = json.Unmarshal([]byte(partitionRows[p.OutputRow-1]), &ex); err != nil {
			return s, nil, err
		}
		if ex.Label < 0 || ex.Label >= len(ex.Options) {
			return s, nil, fmt.Errorf("invalid dataset label")
		}
		task := m.Inputs[p.Input].Source
		if m.Inputs[p.Input].Config != "" {
			task += "/" + m.Inputs[p.Input].Config
		}
		k := task + "\x00" + p.SourceID
		golds[k] = directHash([]byte(ex.Options[ex.Label]))[:16]
		options[k] = map[string]string{}
		for _, text := range ex.Options {
			options[k][directHash([]byte(text))[:16]] = text
		}
	}
	payload, err := os.ReadFile(requests)
	if err != nil {
		return s, nil, err
	}
	if directHash(payload) != s.RequestSHA256 {
		return s, nil, fmt.Errorf("request identity mismatch")
	}
	rows := map[string]directBatchInput{}
	for _, line := range strings.Split(strings.TrimSpace(string(payload)), "\n") {
		var r directBatchInput
		if err = json.Unmarshal([]byte(line), &r); err != nil {
			return s, nil, err
		}
		k := directRowKey(r.Task, r.ID, r.Variant)
		source := r.Task + "\x00" + r.ID
		if _, ok := rows[k]; ok {
			return s, nil, fmt.Errorf("duplicate request")
		}
		if r.ID == "" || r.Task == "" || r.Task == "all" || r.Variant == "" || golds[source] == "" || golds[source] != r.GoldID {
			return s, nil, fmt.Errorf("request not in declared partition or gold changed: %s", r.ID)
		}
		seen := map[string]bool{}
		for _, c := range r.Request.Candidates {
			if seen[c.ID] || options[source][c.ID] != c.Text || c.Text == "" {
				return s, nil, fmt.Errorf("request candidate changed")
			}
			seen[c.ID] = true
		}
		if len(seen) != len(options[source]) {
			return s, nil, fmt.Errorf("request candidate set changed")
		}
		if r.Request.Temperature != 1 {
			return s, nil, fmt.Errorf("study requests must use raw temperature 1")
		}
		rows[k] = r
	}
	if len(rows) != s.Requests || s.Requests < 1 {
		return s, nil, fmt.Errorf("request count mismatch")
	}
	return s, rows, nil
}
