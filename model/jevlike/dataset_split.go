package jevlike

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

const (
	DatasetSplitTrain       = "train"
	DatasetSplitValidation  = "validation"
	DatasetSplitCalibration = "calibration"
	DatasetSplitTest        = "test"
)

// DatasetExample is a validated dataset row with task/source metadata.
type DatasetExample struct {
	Choice        ChoiceExample `json:"choice"`
	SourceID      string        `json:"source_id"`
	GroupID       string        `json:"group_id"`
	Task          string        `json:"task"`
	Genre         string        `json:"genre,omitempty"`
	OriginalSplit string        `json:"original_split,omitempty"`
}

// DatasetSplits groups adapted examples into train/validation/calibration/test.
type DatasetSplits struct {
	Train       []DatasetExample
	Validation  []DatasetExample
	Calibration []DatasetExample
	Test        []DatasetExample
}

// ValidateDatasetChoiceExample applies the stricter dataset import contract.
func ValidateDatasetChoiceExample(example ChoiceExample) (ChoiceExample, error) {
	if err := validateNonBlankUTF8("context", example.Context); err != nil {
		return ChoiceExample{}, err
	}
	if len(example.Options) < 2 || len(example.Options) > 32 {
		return ChoiceExample{}, errors.New("options must contain 2..32 distinct valid UTF-8 nonblank strings")
	}
	options := make([]string, len(example.Options))
	seen := make(map[string]struct{}, len(example.Options))
	for i, option := range example.Options {
		if err := validateNonBlankUTF8("option", option); err != nil {
			return ChoiceExample{}, errors.New("options must contain 2..32 distinct valid UTF-8 nonblank strings")
		}
		if _, ok := seen[option]; ok {
			return ChoiceExample{}, errors.New("options must contain 2..32 distinct valid UTF-8 nonblank strings")
		}
		seen[option] = struct{}{}
		options[i] = option
	}
	if example.Label < 0 || example.Label >= len(options) {
		return ChoiceExample{}, errors.New("label must be an option index")
	}
	return ChoiceExample{Context: example.Context, Options: options, Label: example.Label}, nil
}

// ValidateDatasetExample validates metadata and returns a defensive copy.
func ValidateDatasetExample(example DatasetExample) (DatasetExample, error) {
	choice, err := ValidateDatasetChoiceExample(example.Choice)
	if err != nil {
		return DatasetExample{}, err
	}
	if err := validateNonBlankUTF8("source_id", example.SourceID); err != nil {
		return DatasetExample{}, err
	}
	if err := validateNonBlankUTF8("group_id", example.GroupID); err != nil {
		return DatasetExample{}, err
	}
	if err := validateNonBlankUTF8("task", example.Task); err != nil {
		return DatasetExample{}, err
	}
	if err := validateOptionalUTF8("genre", example.Genre); err != nil {
		return DatasetExample{}, err
	}
	if err := validateOptionalUTF8("original_split", example.OriginalSplit); err != nil {
		return DatasetExample{}, err
	}
	return DatasetExample{
		Choice:        choice,
		SourceID:      example.SourceID,
		GroupID:       example.GroupID,
		Task:          example.Task,
		Genre:         example.Genre,
		OriginalSplit: example.OriginalSplit,
	}, nil
}

// SplitDatasetExamples deterministically assigns train rows to train,
// validation or calibration by source/group hash. Any official non-train rows
// are mapped to the final test split. Duplicate canonical examples are deduped
// within a side, conflicting labels are rejected, and any heldout/train leak is
// surfaced as an error.
func SplitDatasetExamples(source string, examples []DatasetExample, seed int64) (DatasetSplits, error) {
	if err := validateNonBlankUTF8("source", source); err != nil {
		return DatasetSplits{}, err
	}
	if len(examples) == 0 {
		return DatasetSplits{}, errors.New("examples must not be empty")
	}

	type aggregate struct {
		labelText string
		train     *DatasetExample
		heldout   *DatasetExample
	}
	unique := make(map[string]*aggregate, len(examples))
	groupSide := make(map[string]bool)
	groupParent := make(map[string]string)
	var findGroup func(string) string
	findGroup = func(id string) string {
		parent, ok := groupParent[id]
		if !ok {
			groupParent[id] = id
			return id
		}
		if parent != id {
			groupParent[id] = findGroup(parent)
		}
		return groupParent[id]
	}
	unionGroups := func(a, b string) {
		a, b = findGroup(a), findGroup(b)
		if a > b {
			a, b = b, a
		}
		groupParent[b] = a
	}
	for i, example := range examples {
		validated, err := ValidateDatasetExample(example)
		if err != nil {
			return DatasetSplits{}, fmt.Errorf("example %d: %w", i, err)
		}
		// Unknown/missing boundaries must not silently become training data.
		split := normalizeSplitName(validated.OriginalSplit)
		switch split {
		case "train", "validation", "validation_matched", "validation_mismatched", "dev", "test":
		default:
			return DatasetSplits{}, fmt.Errorf("example %d: unknown original split %q", i, validated.OriginalSplit)
		}
		heldout := isOfficialHeldoutSplit(split)
		if previous, ok := groupSide[validated.GroupID]; ok && previous != heldout {
			return DatasetSplits{}, fmt.Errorf("group %q crosses official train/heldout boundary", validated.GroupID)
		}
		groupSide[validated.GroupID] = heldout
		findGroup(validated.GroupID)
		key := canonicalDatasetKey(validated.Choice)
		gold := datasetGoldOption(validated.Choice)
		agg := unique[key]
		if agg == nil {
			agg = &aggregate{labelText: gold}
			unique[key] = agg
		} else if agg.labelText != gold {
			existing := agg.train
			if existing == nil {
				existing = agg.heldout
			}
			return DatasetSplits{}, fmt.Errorf("conflicting labels for canonical example between %q and %q", existing.SourceID, validated.SourceID)
		}
		// Duplicate originals connect their groups: all remaining siblings
		// must share a split, even when duplicate row representatives differ.
		if agg.train != nil {
			unionGroups(agg.train.GroupID, validated.GroupID)
		}
		if agg.heldout != nil {
			unionGroups(agg.heldout.GroupID, validated.GroupID)
		}
		copy := validated
		if heldout {
			if agg.heldout == nil || datasetRepresentativeLess(copy, *agg.heldout) {
				agg.heldout = &copy
			}
		} else {
			if agg.train == nil || datasetRepresentativeLess(copy, *agg.train) {
				agg.train = &copy
			}
		}
	}

	var splits DatasetSplits
	for _, agg := range unique {
		if agg.train != nil && agg.heldout != nil {
			return DatasetSplits{}, fmt.Errorf("duplicate canonical example appears in both official heldout and train: %q and %q", agg.heldout.SourceID, agg.train.SourceID)
		}
		if agg.heldout != nil {
			splits.Test = append(splits.Test, *agg.heldout)
			continue
		}
		switch trainSplitName(source, findGroup(agg.train.GroupID), seed) {
		case DatasetSplitValidation:
			splits.Validation = append(splits.Validation, *agg.train)
		case DatasetSplitCalibration:
			splits.Calibration = append(splits.Calibration, *agg.train)
		default:
			splits.Train = append(splits.Train, *agg.train)
		}
	}

	sortDatasetExamples(splits.Train)
	sortDatasetExamples(splits.Validation)
	sortDatasetExamples(splits.Calibration)
	sortDatasetExamples(splits.Test)
	return splits, nil
}

// PermuteDatasetExample deterministically reorders options and remaps the label.
// The permutation depends on the source, seed and example identity, but not on
// the input option order.
func PermuteDatasetExample(source string, example DatasetExample, seed int64) (DatasetExample, error) {
	if err := validateNonBlankUTF8("source", source); err != nil {
		return DatasetExample{}, err
	}
	validated, err := ValidateDatasetExample(example)
	if err != nil {
		return DatasetExample{}, err
	}
	type rankedOption struct {
		text string
		hash uint64
	}
	ranked := make([]rankedOption, len(validated.Choice.Options))
	for i, option := range validated.Choice.Options {
		ranked[i] = rankedOption{
			text: option,
			hash: stableHash64(
				"dataset-option-permute",
				source,
				strconv.FormatInt(seed, 10),
				validated.SourceID,
				validated.GroupID,
				validated.Choice.Context,
				option,
			),
		}
	}
	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].hash != ranked[j].hash {
			return ranked[i].hash < ranked[j].hash
		}
		return ranked[i].text < ranked[j].text
	})
	gold := datasetGoldOption(validated.Choice)
	options := make([]string, len(ranked))
	label := -1
	for i, option := range ranked {
		options[i] = option.text
		if option.text == gold {
			label = i
		}
	}
	if label < 0 {
		return DatasetExample{}, errors.New("permutation lost gold option")
	}
	validated.Choice.Options = options
	validated.Choice.Label = label
	return validated, nil
}

func canonicalDatasetKey(choice ChoiceExample) string {
	options := append([]string(nil), choice.Options...)
	sort.Strings(options)
	data, _ := json.Marshal(struct {
		Context string   `json:"context"`
		Options []string `json:"options"`
	}{Context: choice.Context, Options: options})
	return string(data)
}

func datasetGoldOption(choice ChoiceExample) string {
	return choice.Options[choice.Label]
}

func trainSplitName(source, groupID string, seed int64) string {
	switch stableHash64("dataset-train-split", source, groupID, strconv.FormatInt(seed, 10)) % 10 {
	case 0:
		return DatasetSplitValidation
	case 1:
		return DatasetSplitCalibration
	default:
		return DatasetSplitTrain
	}
}

func isOfficialHeldoutSplit(split string) bool {
	normalized := normalizeSplitName(split)
	return normalized != "" && normalized != DatasetSplitTrain
}

func normalizeSplitName(split string) string {
	return strings.ToLower(strings.TrimSpace(split))
}

func sortDatasetExamples(examples []DatasetExample) {
	sort.Slice(examples, func(i, j int) bool {
		return datasetRepresentativeLess(examples[i], examples[j])
	})
}

func datasetRepresentativeLess(a, b DatasetExample) bool {
	if a.SourceID != b.SourceID {
		return a.SourceID < b.SourceID
	}
	if a.GroupID != b.GroupID {
		return a.GroupID < b.GroupID
	}
	if a.Task != b.Task {
		return a.Task < b.Task
	}
	if a.Genre != b.Genre {
		return a.Genre < b.Genre
	}
	if normalizeSplitName(a.OriginalSplit) != normalizeSplitName(b.OriginalSplit) {
		return normalizeSplitName(a.OriginalSplit) < normalizeSplitName(b.OriginalSplit)
	}
	if a.Choice.Context != b.Choice.Context {
		return a.Choice.Context < b.Choice.Context
	}
	for i := 0; i < len(a.Choice.Options) && i < len(b.Choice.Options); i++ {
		if a.Choice.Options[i] != b.Choice.Options[i] {
			return a.Choice.Options[i] < b.Choice.Options[i]
		}
	}
	if len(a.Choice.Options) != len(b.Choice.Options) {
		return len(a.Choice.Options) < len(b.Choice.Options)
	}
	return a.Choice.Label < b.Choice.Label
}

func validateNonBlankUTF8(field, value string) error {
	if !utf8.ValidString(value) {
		return fmt.Errorf("%s must be valid UTF-8", field)
	}
	if strings.TrimSpace(value) == "" {
		return fmt.Errorf("%s must be nonblank", field)
	}
	return nil
}

func validateOptionalUTF8(field, value string) error {
	if value == "" {
		return nil
	}
	if !utf8.ValidString(value) {
		return fmt.Errorf("%s must be valid UTF-8", field)
	}
	return nil
}

func stableHash64(parts ...string) uint64 {
	h := sha256.New()
	for _, part := range parts {
		var lenBuf [8]byte
		binary.BigEndian.PutUint64(lenBuf[:], uint64(len(part)))
		_, _ = h.Write(lenBuf[:])
		_, _ = h.Write([]byte(part))
	}
	sum := h.Sum(nil)
	return binary.BigEndian.Uint64(sum[:8])
}

func stableDatasetID(parts ...string) string {
	h := sha256.New()
	for _, part := range parts {
		var lenBuf [8]byte
		binary.BigEndian.PutUint64(lenBuf[:], uint64(len(part)))
		_, _ = h.Write(lenBuf[:])
		_, _ = h.Write([]byte(part))
	}
	sum := h.Sum(nil)
	return hex.EncodeToString(sum[:12])
}
