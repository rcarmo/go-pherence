package jevlike

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"
)

// AdaptDatasetRow converts one supported Hugging Face style row into a
// validated DatasetExample.
func AdaptDatasetRow(source, config string, row json.RawMessage) (DatasetExample, error) {
	if err := validateNonBlankUTF8("source", source); err != nil {
		return DatasetExample{}, err
	}
	if err := validateOptionalUTF8("config", config); err != nil {
		return DatasetExample{}, err
	}
	if !utf8.Valid(row) || !json.Valid(row) {
		return DatasetExample{}, errors.New("row must be valid UTF-8 JSON")
	}
	switch source {
	case "multi_nli":
		return adaptMultiNLI(config, row)
	case "commonsense_qa":
		return adaptCommonsenseQA(config, row)
	case "ai2_arc":
		return adaptAI2ARC(config, row)
	case "clinc_oos":
		return DatasetExample{}, errors.New("clinc_oos requires AdaptCLINC with explicit labels and candidateIDs")
	default:
		return DatasetExample{}, fmt.Errorf("unsupported dataset source %q", source)
	}
}

// AdaptCLINC converts one CLINC-style intent row using an explicit label list
// and candidate subset. The supplied labels define the readable option text.
func AdaptCLINC(row json.RawMessage, labels []string, candidateIDs []int) (DatasetExample, error) {
	if !utf8.Valid(row) || !json.Valid(row) {
		return DatasetExample{}, errors.New("row must be valid UTF-8 JSON")
	}
	options, candidatePos, err := validateCLINCCandidates(labels, candidateIDs)
	if err != nil {
		return DatasetExample{}, err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(row, &fields); err != nil {
		return DatasetExample{}, err
	}
	text, err := requiredOneJSONString(fields, "text", "utterance", "sentence")
	if err != nil {
		return DatasetExample{}, err
	}
	goldID, err := extractCLINCGold(fields, labels)
	if err != nil {
		return DatasetExample{}, err
	}
	label, ok := candidatePos[goldID]
	if !ok {
		return DatasetExample{}, fmt.Errorf("gold CLINC label %d is not present in candidateIDs", goldID)
	}
	sourceID, _, err := optionalJSONString(fields, "id")
	if err != nil {
		return DatasetExample{}, err
	}
	if strings.TrimSpace(sourceID) == "" {
		sourceID = stableDatasetID("clinc_oos", text)
	}
	return ValidateDatasetExample(DatasetExample{
		Choice: ChoiceExample{
			Context: "Task: Choose the intent label that best matches the user utterance.\nUtterance: " + text + "\nAnswer:",
			Options: options,
			Label:   label,
		},
		SourceID: sourceID,
		GroupID:  sourceID,
		Task:     "intent_classification",
	})
}

type rawHFChoices struct {
	Label *[]string `json:"label"`
	Text  *[]string `json:"text"`
}

func adaptMultiNLI(config string, row json.RawMessage) (DatasetExample, error) {
	var raw struct {
		PairID     *string         `json:"pairID"`
		PromptID   json.RawMessage `json:"promptID"`
		Premise    *string         `json:"premise"`
		Hypothesis *string         `json:"hypothesis"`
		Genre      *string         `json:"genre"`
		Label      *int            `json:"label"`
	}
	if err := json.Unmarshal(row, &raw); err != nil {
		return DatasetExample{}, err
	}
	premise, err := requiredStringField("premise", raw.Premise)
	if err != nil {
		return DatasetExample{}, err
	}
	hypothesis, err := requiredStringField("hypothesis", raw.Hypothesis)
	if err != nil {
		return DatasetExample{}, err
	}
	if raw.Label == nil {
		return DatasetExample{}, errors.New("multi_nli label is required")
	}
	if *raw.Label < 0 || *raw.Label > 2 {
		return DatasetExample{}, errors.New("multi_nli label must be 0 entailment, 1 neutral or 2 contradiction")
	}
	sourceID := explicitOrStableID(raw.PairID, "multi_nli", premise, hypothesis)
	groupID := sourceID
	if len(raw.PromptID) > 0 && string(raw.PromptID) != "null" {
		if value, ok, _ := jsonString(raw.PromptID); ok {
			if err := validateNonBlankUTF8("promptID", value); err != nil {
				return DatasetExample{}, err
			}
			groupID = value
		} else if value, ok, _ := jsonInt(raw.PromptID); ok && value >= 0 {
			groupID = strconv.Itoa(value)
		} else {
			return DatasetExample{}, errors.New("promptID must be a nonnegative integer or nonblank string")
		}
	}
	genre := nonBlankOrFallback(raw.Genre, config)
	return ValidateDatasetExample(DatasetExample{
		Choice: ChoiceExample{
			Context: "Task: Decide whether the hypothesis is entailed by, neutral with, or contradicted by the premise.\nPremise: " + premise + "\nHypothesis: " + hypothesis + "\nAnswer:",
			Options: []string{"entailment", "neutral", "contradiction"},
			Label:   *raw.Label,
		},
		SourceID: sourceID,
		GroupID:  groupID,
		Task:     "natural_language_inference",
		Genre:    genre,
	})
}

func adaptCommonsenseQA(config string, row json.RawMessage) (DatasetExample, error) {
	var raw struct {
		ID              *string       `json:"id"`
		Question        *string       `json:"question"`
		QuestionConcept *string       `json:"question_concept"`
		Choices         *rawHFChoices `json:"choices"`
		AnswerKey       *string       `json:"answerKey"`
	}
	if err := json.Unmarshal(row, &raw); err != nil {
		return DatasetExample{}, err
	}
	question, err := requiredStringField("question", raw.Question)
	if err != nil {
		return DatasetExample{}, err
	}
	options, label, err := mapHFChoices(raw.Choices, raw.AnswerKey)
	if err != nil {
		return DatasetExample{}, err
	}
	sourceID := explicitOrStableID(raw.ID, "commonsense_qa", question)
	genre := nonBlankOrFallback(raw.QuestionConcept, config)
	if strings.TrimSpace(genre) == "" {
		genre = "commonsense"
	}
	return ValidateDatasetExample(DatasetExample{
		Choice: ChoiceExample{
			Context: "Task: Choose the best answer to the commonsense question.\nQuestion: " + question + "\nAnswer:",
			Options: options,
			Label:   label,
		},
		SourceID: sourceID,
		GroupID:  sourceID,
		Task:     "multiple_choice_question_answering",
		Genre:    genre,
	})
}

func adaptAI2ARC(config string, row json.RawMessage) (DatasetExample, error) {
	var raw struct {
		ID           *string       `json:"id"`
		Question     *string       `json:"question"`
		QuestionStem *string       `json:"question_stem"`
		Choices      *rawHFChoices `json:"choices"`
		AnswerKey    *string       `json:"answerKey"`
	}
	if err := json.Unmarshal(row, &raw); err != nil {
		return DatasetExample{}, err
	}
	question, err := requiredExistingStringField(raw.Question, raw.QuestionStem)
	if err != nil {
		return DatasetExample{}, errors.New("ai2_arc question is required")
	}
	options, label, err := mapHFChoices(raw.Choices, raw.AnswerKey)
	if err != nil {
		return DatasetExample{}, err
	}
	sourceID := explicitOrStableID(raw.ID, "ai2_arc", question)
	genre := config
	if strings.TrimSpace(genre) == "" {
		genre = "science"
	}
	return ValidateDatasetExample(DatasetExample{
		Choice: ChoiceExample{
			Context: "Task: Choose the best answer to the science question.\nQuestion: " + question + "\nAnswer:",
			Options: options,
			Label:   label,
		},
		SourceID: sourceID,
		GroupID:  sourceID,
		Task:     "multiple_choice_question_answering",
		Genre:    genre,
	})
}

func mapHFChoices(choices *rawHFChoices, answerKey *string) ([]string, int, error) {
	if choices == nil || choices.Label == nil || choices.Text == nil {
		return nil, 0, errors.New("choices.label and choices.text are required")
	}
	labels := append([]string(nil), (*choices.Label)...)
	texts := append([]string(nil), (*choices.Text)...)
	if len(labels) != len(texts) {
		return nil, 0, errors.New("choices.label and choices.text must have the same length")
	}
	if answerKey == nil {
		return nil, 0, errors.New("answerKey is required")
	}
	if err := validateNonBlankUTF8("answerKey", *answerKey); err != nil {
		return nil, 0, err
	}
	seen := make(map[string]struct{}, len(labels))
	options := make([]string, len(texts))
	label := -1
	for i := range labels {
		if err := validateNonBlankUTF8("choice label", labels[i]); err != nil {
			return nil, 0, errors.New("choice labels must be explicit, valid UTF-8 and nonblank")
		}
		if _, ok := seen[labels[i]]; ok {
			return nil, 0, errors.New("choice labels must be unique")
		}
		seen[labels[i]] = struct{}{}
		options[i] = texts[i]
		if labels[i] == *answerKey {
			label = i
		}
	}
	if label < 0 {
		return nil, 0, fmt.Errorf("answerKey %q is not present in choices.label", *answerKey)
	}
	return options, label, nil
}

func validateCLINCCandidates(labels []string, candidateIDs []int) ([]string, map[int]int, error) {
	if len(candidateIDs) < 2 || len(candidateIDs) > 32 {
		return nil, nil, errors.New("candidateIDs must contain 2..32 distinct indexes")
	}
	for i, label := range labels {
		if err := validateNonBlankUTF8(fmt.Sprintf("labels[%d]", i), label); err != nil {
			return nil, nil, err
		}
	}
	seen := make(map[int]struct{}, len(candidateIDs))
	positions := make(map[int]int, len(candidateIDs))
	options := make([]string, len(candidateIDs))
	for i, id := range candidateIDs {
		if id < 0 || id >= len(labels) {
			return nil, nil, fmt.Errorf("candidateIDs[%d] is out of range", i)
		}
		if _, ok := seen[id]; ok {
			return nil, nil, errors.New("candidateIDs must be distinct")
		}
		seen[id] = struct{}{}
		positions[id] = i
		options[i] = labels[id]
	}
	return options, positions, nil
}

func extractCLINCGold(fields map[string]json.RawMessage, labels []string) (int, error) {
	for _, key := range []string{"label", "intent", "intent_label"} {
		raw, ok := fields[key]
		if !ok {
			continue
		}
		if id, ok, err := jsonInt(raw); err != nil {
			return 0, fmt.Errorf("%s: %w", key, err)
		} else if ok {
			return id, nil
		}
		if value, ok, err := jsonString(raw); err != nil {
			return 0, fmt.Errorf("%s: %w", key, err)
		} else if ok {
			if n, err := strconv.Atoi(value); err == nil {
				return n, nil
			}
			if idx := indexOf(labels, value); idx >= 0 {
				return idx, nil
			}
			return 0, fmt.Errorf("%s: label %q is not present in labels", key, value)
		}
	}
	for _, key := range []string{"intent_name", "label_text"} {
		raw, ok := fields[key]
		if !ok {
			continue
		}
		value, ok, err := jsonString(raw)
		if err != nil {
			return 0, fmt.Errorf("%s: %w", key, err)
		}
		if ok {
			if idx := indexOf(labels, value); idx >= 0 {
				return idx, nil
			}
			return 0, fmt.Errorf("%s: label %q is not present in labels", key, value)
		}
	}
	return 0, errors.New("missing CLINC gold label")
}

func requiredStringField(name string, value *string) (string, error) {
	if value == nil {
		return "", fmt.Errorf("%s is required", name)
	}
	if err := validateNonBlankUTF8(name, *value); err != nil {
		return "", err
	}
	return *value, nil
}

func requiredExistingStringField(values ...*string) (string, error) {
	for _, value := range values {
		if value == nil {
			continue
		}
		if err := validateNonBlankUTF8("string field", *value); err != nil {
			if strings.TrimSpace(*value) == "" {
				continue
			}
			return "", err
		}
		return *value, nil
	}
	return "", errors.New("missing string field")
}

func explicitOrStableID(value *string, parts ...string) string {
	if value != nil && strings.TrimSpace(*value) != "" {
		return *value
	}
	return stableDatasetID(parts...)
}

func nonBlankOrFallback(value *string, fallback string) string {
	if value != nil && strings.TrimSpace(*value) != "" {
		return *value
	}
	return fallback
}

func requiredOneJSONString(fields map[string]json.RawMessage, keys ...string) (string, error) {
	for _, key := range keys {
		value, ok, err := optionalJSONString(fields, key)
		if err != nil {
			return "", err
		}
		if ok && strings.TrimSpace(value) != "" {
			return value, nil
		}
	}
	return "", fmt.Errorf("missing one of %s", strings.Join(keys, ", "))
}

func optionalJSONString(fields map[string]json.RawMessage, key string) (string, bool, error) {
	raw, ok := fields[key]
	if !ok {
		return "", false, nil
	}
	value, ok, err := jsonString(raw)
	if err != nil {
		return "", false, fmt.Errorf("%s: %w", key, err)
	}
	if !ok {
		return "", false, nil
	}
	if err := validateOptionalUTF8(key, value); err != nil {
		return "", false, err
	}
	return value, true, nil
}

func jsonString(raw json.RawMessage) (string, bool, error) {
	if string(raw) == "null" {
		return "", false, nil
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", false, nil
	}
	return value, true, nil
}

func jsonInt(raw json.RawMessage) (int, bool, error) {
	if string(raw) == "null" {
		return 0, false, nil
	}
	var value int
	if err := json.Unmarshal(raw, &value); err != nil {
		return 0, false, nil
	}
	return value, true, nil
}
