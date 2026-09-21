// Copyright (c) 2026 Minimal Labs
// Copyright (c) 2026 Rui Carmo (Go adaptation)
// SPDX-License-Identifier: MIT

// Package jevlike provides validated JSONL loading, byte-level batching and
// small dataset builders for JEV-like multiple-choice tasks.
package jevlike

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"net/url"
	"os"
	"path/filepath"
	"strings"
)

const (
	defaultWikispeediaMaxOptions       = 64
	wikispeediaArticleChars            = 2048
	syntheticStride              int64 = 104729
)

var (
	colours    = []string{"amber", "azure", "bronze", "coral", "crimson", "gold", "green", "indigo"}
	animals    = []string{"badger", "crane", "dolphin", "falcon", "gecko", "heron", "ibis", "jaguar"}
	directions = []string{"north", "south", "east", "west"}
)

// ChoiceExample is a validated multiple-choice example.
type ChoiceExample struct {
	Context string   `json:"context"`
	Options []string `json:"options"`
	Label   int      `json:"label"`
}

// SplitSizes names the standard train/validation/test split sizes or counts.
type SplitSizes struct {
	Train      int
	Validation int
	Test       int
}

// ByteBatch is a padded byte-token batch.
//
// Token ids are byte values plus one, leaving zero as the pad id.
// ContextMask marks non-pad context tokens, OptionTokenMask marks non-pad option
// tokens and OptionMask marks which option slots are present in each row.
type ByteBatch struct {
	ContextIDs      [][]uint16
	ContextMask     [][]bool
	OptionIDs       [][][]uint16
	OptionTokenMask [][][]bool
	OptionMask      [][]bool
	Labels          []int
}

// WikispeediaOptions controls BuildWikispeedia.
type WikispeediaOptions struct {
	// MaxOptions caps the number of answer options per example. Zero uses the
	// reference default of 64.
	MaxOptions int
}

type rawChoiceExample struct {
	Context *string   `json:"context"`
	Options *[]string `json:"options"`
	Label   *int      `json:"label"`
}

// ValidateChoiceExample validates a typed example and returns a defensive copy.
func ValidateChoiceExample(example ChoiceExample) (ChoiceExample, error) {
	if len(example.Options) < 2 {
		return ChoiceExample{}, errors.New("options must contain at least two non-empty strings")
	}
	options := make([]string, len(example.Options))
	for i, option := range example.Options {
		if option == "" {
			return ChoiceExample{}, errors.New("options must contain at least two non-empty strings")
		}
		options[i] = option
	}
	if example.Label < 0 || example.Label >= len(options) {
		return ChoiceExample{}, errors.New("label must be an option index")
	}
	return ChoiceExample{Context: example.Context, Options: options, Label: example.Label}, nil
}

// ParseChoiceExampleJSON validates one JSON object line.
func ParseChoiceExampleJSON(data []byte) (ChoiceExample, error) {
	var raw rawChoiceExample
	if err := json.Unmarshal(data, &raw); err != nil {
		return ChoiceExample{}, err
	}
	if raw.Context == nil || raw.Options == nil {
		return ChoiceExample{}, errors.New("each row needs string context and list options")
	}
	if raw.Label == nil {
		return ChoiceExample{}, errors.New("label must be an option index")
	}
	return ValidateChoiceExample(ChoiceExample{
		Context: *raw.Context,
		Options: append([]string(nil), (*raw.Options)...),
		Label:   *raw.Label,
	})
}

// LoadJSONL loads validated examples from a UTF-8 JSONL file.
func LoadJSONL(path string) ([]ChoiceExample, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	var examples []ChoiceExample
	for lineNo := 1; scanner.Scan(); lineNo++ {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		example, err := ParseChoiceExampleJSON([]byte(line))
		if err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		}
		examples = append(examples, example)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(examples) == 0 {
		return nil, fmt.Errorf("no examples in %s", path)
	}
	return examples, nil
}

// WriteJSONL writes validated examples to path.
func WriteJSONL(path string, examples []ChoiceExample) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	for i, example := range examples {
		validated, err := ValidateChoiceExample(example)
		if err != nil {
			return fmt.Errorf("%s example %d: %w", path, i, err)
		}
		line, err := json.Marshal(validated)
		if err != nil {
			return err
		}
		if _, err := writer.Write(line); err != nil {
			return err
		}
		if err := writer.WriteByte('\n'); err != nil {
			return err
		}
	}
	return writer.Flush()
}

// EncodeBytes encodes text as UTF-8, truncates by byte count and shifts each
// byte by one so zero remains available as the pad id.
func EncodeBytes(text string, maxBytes int) ([]uint16, error) {
	if maxBytes < 0 {
		return nil, fmt.Errorf("max bytes must be non-negative")
	}
	encoded := []byte(text)
	if len(encoded) > maxBytes {
		encoded = encoded[:maxBytes]
	}
	ids := make([]uint16, len(encoded))
	for i, value := range encoded {
		ids[i] = uint16(value) + 1
	}
	return ids, nil
}

// BuildByteBatch builds a padded byte-token batch.
func BuildByteBatch(examples []ChoiceExample, contextBytes, optionBytes int) (ByteBatch, error) {
	if len(examples) == 0 {
		return ByteBatch{}, errors.New("batch needs at least one example")
	}
	if contextBytes < 0 || optionBytes < 0 {
		return ByteBatch{}, errors.New("token limits must be non-negative")
	}

	validated := make([]ChoiceExample, len(examples))
	contexts := make([][]uint16, len(examples))
	optionRows := make([][][]uint16, len(examples))
	maxContext, maxOptions, maxOptionTokens := 0, 0, 0
	for i, example := range examples {
		item, err := ValidateChoiceExample(example)
		if err != nil {
			return ByteBatch{}, fmt.Errorf("example %d: %w", i, err)
		}
		validated[i] = item
		contextIDs, err := EncodeBytes(item.Context, contextBytes)
		if err != nil {
			return ByteBatch{}, err
		}
		contexts[i] = contextIDs
		if len(contextIDs) > maxContext {
			maxContext = len(contextIDs)
		}
		if len(item.Options) > maxOptions {
			maxOptions = len(item.Options)
		}
		row := make([][]uint16, len(item.Options))
		for j, option := range item.Options {
			optionIDs, err := EncodeBytes(option, optionBytes)
			if err != nil {
				return ByteBatch{}, err
			}
			row[j] = optionIDs
			if len(optionIDs) > maxOptionTokens {
				maxOptionTokens = len(optionIDs)
			}
		}
		optionRows[i] = row
	}

	batch := ByteBatch{
		ContextIDs:      makeUint16Matrix(len(validated), maxContext),
		ContextMask:     makeBoolMatrix(len(validated), maxContext),
		OptionIDs:       makeUint16Cube(len(validated), maxOptions, maxOptionTokens),
		OptionTokenMask: makeBoolCube(len(validated), maxOptions, maxOptionTokens),
		OptionMask:      makeBoolMatrix(len(validated), maxOptions),
		Labels:          make([]int, len(validated)),
	}
	for i, example := range validated {
		copy(batch.ContextIDs[i], contexts[i])
		for j := range contexts[i] {
			batch.ContextMask[i][j] = true
		}
		batch.Labels[i] = example.Label
		for j, tokens := range optionRows[i] {
			batch.OptionMask[i][j] = true
			copy(batch.OptionIDs[i][j], tokens)
			for k := range tokens {
				batch.OptionTokenMask[i][j][k] = true
			}
		}
	}
	return batch, nil
}

// SyntheticExample deterministically builds a synthetic example from a Go RNG
// seed. It does not claim byte-for-byte parity with the Python reference RNG.
func SyntheticExample(seed int64) ChoiceExample {
	rng := rand.New(rand.NewSource(seed))
	target := fmt.Sprintf("%s %s", colours[rng.Intn(len(colours))], animals[rng.Intn(len(animals))])
	count := 2 + rng.Intn(7)
	options := []string{target}
	seen := map[string]struct{}{target: {}}
	for len(options) < count {
		candidate := fmt.Sprintf("%s %s", colours[rng.Intn(len(colours))], animals[rng.Intn(len(animals))])
		if _, ok := seen[candidate]; ok {
			continue
		}
		seen[candidate] = struct{}{}
		options = append(options, candidate)
	}
	rng.Shuffle(len(options), func(i, j int) {
		options[i], options[j] = options[j], options[i]
	})
	notes := make([]string, 8)
	for i := range notes {
		notes[i] = directions[rng.Intn(len(directions))]
	}
	context := fmt.Sprintf("Choose the exact badge %s. Notes: %s. Badge: %s.", target, strings.Join(notes, " "), target)
	return ChoiceExample{Context: context, Options: options, Label: indexOf(options, target)}
}

// WriteSyntheticDataset writes train/validation/test synthetic JSONL files.
func WriteSyntheticDataset(outputDir string, sizes SplitSizes, seed int64) error {
	if sizes.Train < 0 || sizes.Validation < 0 || sizes.Test < 0 {
		return errors.New("split sizes must be non-negative")
	}
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return err
	}
	entries := []struct {
		name string
		size int
	}{
		{name: "train", size: sizes.Train},
		{name: "validation", size: sizes.Validation},
		{name: "test", size: sizes.Test},
	}
	offset := int64(0)
	for _, entry := range entries {
		path := filepath.Join(outputDir, entry.name+".jsonl")
		file, err := os.Create(path)
		if err != nil {
			return err
		}
		writer := bufio.NewWriter(file)
		for i := 0; i < entry.size; i++ {
			example, err := ValidateChoiceExample(SyntheticExample(seed + offset + int64(i)*syntheticStride))
			if err != nil {
				file.Close()
				return err
			}
			line, err := json.Marshal(example)
			if err != nil {
				file.Close()
				return err
			}
			if _, err := writer.Write(line); err != nil {
				file.Close()
				return err
			}
			if err := writer.WriteByte('\n'); err != nil {
				file.Close()
				return err
			}
		}
		if err := writer.Flush(); err != nil {
			file.Close()
			return err
		}
		if err := file.Close(); err != nil {
			return err
		}
		offset += int64(entry.size) * syntheticStride
	}
	return nil
}

// BuildWikispeedia builds train/validation/test JSONL files from the classic
// Wikispeedia graph and path dump.
func BuildWikispeedia(rootDir, outputDir string, opts WikispeediaOptions) (SplitSizes, error) {
	maxOptions := opts.MaxOptions
	if maxOptions == 0 {
		maxOptions = defaultWikispeediaMaxOptions
	}
	if maxOptions < 2 {
		return SplitSizes{}, errors.New("max options must be at least 2")
	}

	graphDir := filepath.Join(rootDir, "wikispeedia_paths-and-graph")
	outgoing, err := readOutgoingLinks(filepath.Join(graphDir, "links.tsv"))
	if err != nil {
		return SplitSizes{}, err
	}
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return SplitSizes{}, err
	}

	trainFile, err := os.Create(filepath.Join(outputDir, "train.jsonl"))
	if err != nil {
		return SplitSizes{}, err
	}
	defer trainFile.Close()
	validationFile, err := os.Create(filepath.Join(outputDir, "validation.jsonl"))
	if err != nil {
		return SplitSizes{}, err
	}
	defer validationFile.Close()
	testFile, err := os.Create(filepath.Join(outputDir, "test.jsonl"))
	if err != nil {
		return SplitSizes{}, err
	}
	defer testFile.Close()

	writers := map[string]*bufio.Writer{
		"train":      bufio.NewWriter(trainFile),
		"validation": bufio.NewWriter(validationFile),
		"test":       bufio.NewWriter(testFile),
	}
	counts := SplitSizes{}
	flush := func() error {
		for _, writer := range writers {
			if err := writer.Flush(); err != nil {
				return err
			}
		}
		return nil
	}

	pathsFile, err := os.Open(filepath.Join(graphDir, "paths_finished.tsv"))
	if err != nil {
		return SplitSizes{}, err
	}
	defer pathsFile.Close()

	scanner := bufio.NewScanner(pathsFile)
	scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	for row := 0; scanner.Scan(); row++ {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Split(line, "\t")
		if len(fields) < 4 {
			return SplitSizes{}, fmt.Errorf("malformed paths_finished.tsv row %d", row+1)
		}
		path := unwindPath(fields[3])
		if len(path) < 2 {
			continue
		}
		step := int(stableKey(fmt.Sprintf("%s:%s:%d", fields[0], fields[1], row)) % uint64(len(path)-1))
		current, click, target := path[step], path[step+1], path[len(path)-1]
		candidates := uniqueStrings(outgoing[current])
		if len(candidates) < 2 || indexOf(candidates, click) < 0 {
			continue
		}
		others := make([]string, 0, len(candidates)-1)
		for _, candidate := range candidates {
			if candidate != click {
				others = append(others, candidate)
			}
		}
		rng := rand.New(rand.NewSource(int64(stableKey(fmt.Sprintf("%d:%s:menu", row, target)))))
		rng.Shuffle(len(others), func(i, j int) {
			others[i], others[j] = others[j], others[i]
		})
		menu := append([]string{click}, others[:minInt(len(others), maxOptions-1)]...)
		rng.Shuffle(len(menu), func(i, j int) {
			menu[i], menu[j] = menu[j], menu[i]
		})

		articleBytes, err := os.ReadFile(filepath.Join(rootDir, "plaintext_articles", current+".txt"))
		if err != nil {
			return SplitSizes{}, err
		}
		body := normalizeArticleText(articleBytes)
		example, err := ValidateChoiceExample(ChoiceExample{
			Context: fmt.Sprintf("Target article: %s\nCurrent article: %s\n%s", titleText(target), titleText(current), truncateRunes(body, wikispeediaArticleChars)),
			Options: mapStrings(menu, titleText),
			Label:   indexOf(menu, click),
		})
		if err != nil {
			return SplitSizes{}, err
		}
		split := splitName(target)
		lineBytes, err := json.Marshal(example)
		if err != nil {
			return SplitSizes{}, err
		}
		if _, err := writers[split].Write(lineBytes); err != nil {
			return SplitSizes{}, err
		}
		if err := writers[split].WriteByte('\n'); err != nil {
			return SplitSizes{}, err
		}
		switch split {
		case "train":
			counts.Train++
		case "validation":
			counts.Validation++
		case "test":
			counts.Test++
		}
	}
	if err := scanner.Err(); err != nil {
		return SplitSizes{}, err
	}
	if err := flush(); err != nil {
		return SplitSizes{}, err
	}
	return counts, nil
}

func readOutgoingLinks(path string) (map[string][]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	outgoing := make(map[string][]string)
	for lineNo := 1; scanner.Scan(); lineNo++ {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Split(line, "\t")
		if len(fields) != 2 {
			return nil, fmt.Errorf("%s:%d: malformed link row", path, lineNo)
		}
		outgoing[fields[0]] = append(outgoing[fields[0]], fields[1])
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return outgoing, nil
}

func unwindPath(raw string) []string {
	parts := strings.Split(raw, ";")
	path := make([]string, 0, len(parts))
	for _, part := range parts {
		switch part {
		case "<":
			if len(path) > 1 {
				path = path[:len(path)-1]
			}
		case "":
			// Keep parity with the reference: empty nodes simply append as-is.
			path = append(path, part)
		default:
			path = append(path, part)
		}
	}
	return path
}

func splitName(target string) string {
	switch stableKey(target+":split") % 10 {
	case 0:
		return "test"
	case 1:
		return "validation"
	default:
		return "train"
	}
}

func stableKey(text string) uint64 {
	sum := sha256.Sum256([]byte(text))
	return binary.BigEndian.Uint64(sum[:8])
}

func titleText(text string) string {
	decoded, err := url.PathUnescape(text)
	if err != nil {
		decoded = text
	}
	return strings.ReplaceAll(decoded, "_", " ")
}

func normalizeArticleText(data []byte) string {
	valid := bytes.ToValidUTF8(data, []byte("\uFFFD"))
	return strings.Join(strings.Fields(string(valid)), " ")
}

func truncateRunes(text string, limit int) string {
	if limit < 0 {
		return ""
	}
	runes := []rune(text)
	if len(runes) <= limit {
		return text
	}
	return string(runes[:limit])
}

func mapStrings(values []string, fn func(string) string) []string {
	mapped := make([]string, len(values))
	for i, value := range values {
		mapped[i] = fn(value)
	}
	return mapped
}

func uniqueStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(values))
	unique := make([]string, 0, len(values))
	for _, value := range values {
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		unique = append(unique, value)
	}
	return unique
}

func indexOf(values []string, target string) int {
	for i, value := range values {
		if value == target {
			return i
		}
	}
	return -1
}

func makeUint16Matrix(rows, cols int) [][]uint16 {
	matrix := make([][]uint16, rows)
	for i := range matrix {
		matrix[i] = make([]uint16, cols)
	}
	return matrix
}

func makeBoolMatrix(rows, cols int) [][]bool {
	matrix := make([][]bool, rows)
	for i := range matrix {
		matrix[i] = make([]bool, cols)
	}
	return matrix
}

func makeUint16Cube(a, b, c int) [][][]uint16 {
	cube := make([][][]uint16, a)
	for i := range cube {
		cube[i] = make([][]uint16, b)
		for j := range cube[i] {
			cube[i][j] = make([]uint16, c)
		}
	}
	return cube
}

func makeBoolCube(a, b, c int) [][][]bool {
	cube := make([][][]bool, a)
	for i := range cube {
		cube[i] = make([][]bool, b)
		for j := range cube[i] {
			cube[i][j] = make([]bool, c)
		}
	}
	return cube
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
