package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
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

// Input metadata is kept outside model input. Every local export must be pinned
// and hashed, including the ordered CLINC labels/candidate policy if used.
type datasetInput struct {
	Source         string   `json:"source"`
	Config         string   `json:"config"`
	Revision       string   `json:"revision"`
	License        string   `json:"license"`
	Split          string   `json:"split"`
	Path           string   `json:"path"`
	SHA256         string   `json:"sha256"`
	Labels         []string `json:"labels,omitempty"`
	CandidateIDs   []int    `json:"candidate_ids,omitempty"`
	CandidateCount int      `json:"candidate_count,omitempty"`
	CandidateSeed  int64    `json:"candidate_seed,omitempty"`
	OOSID          *int     `json:"oos_id,omitempty"`
}
type datasetPrepareSpec struct {
	Version int            `json:"version"`
	Inputs  []datasetInput `json:"inputs"`
}
type datasetRowProvenance struct {
	Input         int      `json:"input"`
	Line          int      `json:"line"`
	SourceID      string   `json:"source_id"`
	GroupID       string   `json:"group_id"`
	Task          string   `json:"task"`
	Genre         string   `json:"genre,omitempty"`
	OriginalSplit string   `json:"original_split"`
	ContentSHA256 string   `json:"content_sha256"`
	Partition     string   `json:"partition"`
	Options       []string `json:"options"`
	OutputRow     int      `json:"output_row"`
}
type preparedArtifact struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Rows   int    `json:"rows"`
}

func runPrepare(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("prepare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifest := fs.String("manifest", "", "Pinned local JSONL source manifest; no network access")
	out := fs.String("output-dir", "", "New directory for four splits and provenance (must not exist)")
	seed := fs.Int64("seed", 7, "Deterministic group split and option-order seed")
	if err := parseSubcommandFlags(fs, args); err != nil {
		if err == errHelpRequested {
			return nil
		}
		return err
	}
	if *manifest == "" || *out == "" {
		return fmt.Errorf("-manifest and -output-dir are required")
	}
	if _, err := os.Lstat(*out); !os.IsNotExist(err) {
		return fmt.Errorf("output directory must not exist: %s", *out)
	}
	data, err := os.ReadFile(*manifest)
	if err != nil {
		return err
	}
	var spec datasetPrepareSpec
	dec := json.NewDecoder(strings.NewReader(string(data)))
	dec.DisallowUnknownFields()
	if err = dec.Decode(&spec); err != nil {
		return err
	}
	if dec.Decode(new(any)) != io.EOF {
		return fmt.Errorf("trailing manifest data")
	}
	if spec.Version != 1 || len(spec.Inputs) == 0 {
		return fmt.Errorf("version 1 and nonempty inputs required")
	}
	var examples []jevlike.DatasetExample
	var provenance []datasetRowProvenance
	identities := map[string]string{}
	for inputIdx, input := range spec.Inputs {
		if err := validateDatasetInput(input); err != nil {
			return fmt.Errorf("input %d: %w", inputIdx, err)
		}
		path := input.Path
		if !filepath.IsAbs(path) {
			path = filepath.Join(filepath.Dir(*manifest), path)
		}
		rows, err := readDatasetInput(path, input, inputIdx, identities)
		if err != nil {
			return err
		}
		for _, row := range rows {
			examples = append(examples, row.example)
			provenance = append(provenance, row.provenance)
		}
	}
	splits, err := jevlike.SplitDatasetExamples("jevlike-pilot-v1", examples, *seed)
	if err != nil {
		return err
	}
	sets := map[string][]jevlike.DatasetExample{"train": splits.Train, "validation": splits.Validation, "calibration": splits.Calibration, "test": splits.Test}
	partitions := map[string]string{}
	outputRows := map[string]int{}
	for name, set := range sets {
		for row, ex := range set {
			key := datasetContentHash(ex.Choice)
			partitions[key] = name
			outputRows[key] = row + 1
		}
	}
	for i := range provenance {
		provenance[i].Partition = partitions[provenance[i].ContentSHA256]
		provenance[i].OutputRow = outputRows[provenance[i].ContentSHA256]
	}
	// Stable provenance order doesn't depend on file input order.
	sort.Slice(provenance, func(i, j int) bool { return provenance[i].SourceID < provenance[j].SourceID })
	parent := filepath.Dir(*out)
	if err = os.MkdirAll(parent, 0o755); err != nil {
		return err
	}
	tmp, err := os.MkdirTemp(parent, ".jevlike-prepare-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	var artifacts []preparedArtifact
	for _, name := range []string{"train", "validation", "calibration", "test"} {
		var choices []jevlike.ChoiceExample
		for _, ex := range sets[name] {
			permuted, err := jevlike.PermuteDatasetExample("jevlike-pilot-v1", ex, *seed)
			if err != nil {
				return err
			}
			choices = append(choices, permuted.Choice)
		}
		a, err := writePreparedJSONL(tmp, name+".jsonl", choices)
		if err != nil {
			return err
		}
		artifacts = append(artifacts, a)
	}
	a, err := writePreparedJSONL(tmp, "provenance.jsonl", provenance)
	if err != nil {
		return err
	}
	artifacts = append(artifacts, a)
	specHash := sha256.Sum256(data)
	report := struct {
		Version       int                `json:"version"`
		Preprocessing string             `json:"preprocessing"`
		Seed          int64              `json:"seed"`
		SpecSHA256    string             `json:"spec_sha256"`
		Inputs        []datasetInput     `json:"inputs"`
		InputRows     int                `json:"input_rows"`
		Duplicates    int                `json:"deduplicated_rows"`
		Artifacts     []preparedArtifact `json:"artifacts"`
	}{1, "jevlike-dataset-v1:exact-orderless-dedup;group-hash-80/10/10;official-heldout-test", *seed, hex.EncodeToString(specHash[:]), spec.Inputs, len(examples), len(examples) - len(partitions), artifacts}
	encoded, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(filepath.Join(tmp, "manifest.json"), append(encoded, '\n'), 0o644); err != nil {
		return err
	}
	if _, err = os.Lstat(*out); !os.IsNotExist(err) {
		return fmt.Errorf("output appeared during preparation: %s", *out)
	}
	if err = os.Rename(tmp, *out); err != nil {
		return err
	}
	return writeJSON(stdout, report)
}

func validateDatasetInput(in datasetInput) error {
	if in.Source == "" || in.Path == "" || strings.TrimSpace(in.License) == "" {
		return fmt.Errorf("source, path and licence required")
	}
	if len(in.Revision) != 40 {
		return fmt.Errorf("revision must be a full 40-character source commit")
	}
	if _, err := hex.DecodeString(in.Revision); err != nil {
		return fmt.Errorf("invalid revision")
	}
	if len(in.SHA256) != 64 {
		return fmt.Errorf("sha256 must be 64 hex characters")
	}
	if _, err := hex.DecodeString(in.SHA256); err != nil {
		return fmt.Errorf("invalid sha256")
	}
	switch in.Split {
	case "train", "validation", "validation_matched", "validation_mismatched", "dev", "test":
	default:
		return fmt.Errorf("unknown original split %q", in.Split)
	}
	return nil
}

type preparedRow struct {
	example    jevlike.DatasetExample
	provenance datasetRowProvenance
}

func readDatasetInput(path string, in datasetInput, inputIdx int, identities map[string]string) ([]preparedRow, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	h := sha256.New()
	scan := bufio.NewScanner(io.TeeReader(f, h))
	scan.Buffer(make([]byte, 64*1024), 8*1024*1024)
	var rows []preparedRow
	for line := 1; scan.Scan(); line++ {
		if len(strings.TrimSpace(scan.Text())) == 0 {
			continue
		}
		var ex jevlike.DatasetExample
		if in.Source == "clinc_oos" {
			ids := in.CandidateIDs
			if in.CandidateCount != 0 {
				if len(ids) != 0 || in.OOSID == nil {
					return nil, fmt.Errorf("CLINC automatic candidates require explicit OOS ID and no fixed candidate list")
				}
				ids, err = jevlike.CLINCCandidates(scan.Bytes(), in.Labels, in.CandidateCount, in.CandidateSeed, *in.OOSID)
				if err != nil {
					return nil, fmt.Errorf("%s:%d: %w", path, line, err)
				}
			}
			ex, err = jevlike.AdaptCLINC(scan.Bytes(), in.Labels, ids)
		} else {
			ex, err = jevlike.AdaptDatasetRow(in.Source, in.Config, scan.Bytes())
		}
		if err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, line, err)
		}
		ns := in.Source + "/" + in.Config + "/"
		ex.SourceID = ns + ex.SourceID
		ex.GroupID = ns + ex.GroupID
		ex.OriginalSplit = in.Split
		key := datasetContentHash(ex.Choice)
		gold := ex.Choice.Options[ex.Choice.Label]
		identityKey := key + "\x00" + gold
		if previous, ok := identities[ex.SourceID]; ok && previous != identityKey {
			return nil, fmt.Errorf("source row ID %q has conflicting content or label", ex.SourceID)
		}
		identities[ex.SourceID] = identityKey
		rows = append(rows, preparedRow{ex, datasetRowProvenance{inputIdx, line, ex.SourceID, ex.GroupID, ex.Task, ex.Genre, in.Split, key, "", append([]string(nil), ex.Choice.Options...), 0}})
	}
	if err = scan.Err(); err != nil {
		return nil, err
	}
	if hex.EncodeToString(h.Sum(nil)) != strings.ToLower(in.SHA256) {
		return nil, fmt.Errorf("%s: source SHA256 mismatch", path)
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("%s: no labelled rows", path)
	}
	return rows, nil
}
func datasetContentHash(ex jevlike.ChoiceExample) string {
	opts := append([]string(nil), ex.Options...)
	sort.Strings(opts)
	b, _ := json.Marshal(struct {
		Context string
		Options []string
	}{ex.Context, opts})
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}
func writePreparedJSONL[T any](dir, name string, rows []T) (preparedArtifact, error) {
	f, err := os.OpenFile(filepath.Join(dir, name), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return preparedArtifact{}, err
	}
	h := sha256.New()
	enc := json.NewEncoder(io.MultiWriter(f, h))
	for _, row := range rows {
		if err = enc.Encode(row); err != nil {
			f.Close()
			return preparedArtifact{}, err
		}
	}
	if err = f.Sync(); err != nil {
		f.Close()
		return preparedArtifact{}, err
	}
	if err = f.Close(); err != nil {
		return preparedArtifact{}, err
	}
	return preparedArtifact{name, hex.EncodeToString(h.Sum(nil)), len(rows)}, nil
}
