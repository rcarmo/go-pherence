package pockettts

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

const (
	// Upstream prompt/target splitting requires at least one second of audio on
	// either side of an aligned word boundary.
	TrainingMinimumCutSeconds = 1.0
	trainingManifestLineLimit = 8 << 20
)

// TrainingWord is one utterance-relative word alignment from an aligned
// Pocket TTS JSONL manifest. Pointer timestamps preserve JSON null so malformed
// or incomplete alignments cannot silently become zero-second boundaries.
type TrainingWord struct {
	Word  string   `json:"word"`
	Start *float64 `json:"start"`
	End   *float64 `json:"end"`
}

// TrainingEntry is the native subset of the upstream training manifest. Extra
// JSON fields such as speaker and audio_filepath are intentionally ignored.
type TrainingEntry struct {
	Path        string         `json:"path"`
	Duration    float64        `json:"duration"`
	Transcript  string         `json:"transcript"`
	Words       []TrainingWord `json:"words"`
	Start       float64        `json:"start,omitempty"`
	LatentsFile string         `json:"latents_file,omitempty"`
}

// TrainingCut identifies an eligible prompt/target split. WordIndex is the
// first target word, matching upstream's " ".join(words[i:]) behavior.
type TrainingCut struct {
	Seconds          float64
	WordIndex        int
	TargetTranscript string
}

func finite64(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// EligibleTrainingCuts returns aligned boundaries that leave minSide seconds
// on both sides. A non-positive minSide selects the upstream one-second rule.
func (e TrainingEntry) EligibleTrainingCuts(minSide float64) []TrainingCut {
	if minSide <= 0 {
		minSide = TrainingMinimumCutSeconds
	}
	capacity := len(e.Words) - 1
	if capacity < 0 {
		capacity = 0
	}
	cuts := make([]TrainingCut, 0, capacity)
	for i := 1; i < len(e.Words); i++ {
		prev, current := e.Words[i-1], e.Words[i]
		if prev.End == nil || current.Start == nil {
			continue
		}
		cut := (*prev.End + *current.Start) / 2
		if cut < minSide || e.Duration-cut < minSide {
			continue
		}
		parts := make([]string, 0, len(e.Words)-i)
		for _, word := range e.Words[i:] {
			parts = append(parts, word.Word)
		}
		cuts = append(cuts, TrainingCut{Seconds: cut, WordIndex: i, TargetTranscript: strings.Join(parts, " ")})
	}
	return cuts
}

// Validate enforces the aligned-manifest boundary needed by native prompt
// splitting. It does not touch audio or latent files; file existence is a
// separate data-materialization concern.
func (e TrainingEntry) Validate() error {
	if strings.TrimSpace(e.Path) == "" {
		return fmt.Errorf("path is empty")
	}
	if !finite64(e.Duration) || e.Duration <= 0 {
		return fmt.Errorf("duration must be positive and finite")
	}
	if !finite64(e.Start) || e.Start < 0 {
		return fmt.Errorf("start must be non-negative and finite")
	}
	if strings.TrimSpace(e.Transcript) == "" {
		return fmt.Errorf("transcript is empty")
	}
	if len(e.Words) < 2 {
		return fmt.Errorf("at least two aligned words are required")
	}
	lastStart, lastEnd := -1.0, -1.0
	for i, word := range e.Words {
		if strings.TrimSpace(word.Word) == "" {
			return fmt.Errorf("word %d is empty", i)
		}
		if word.Start == nil || word.End == nil {
			return fmt.Errorf("word %d has null alignment", i)
		}
		start, end := *word.Start, *word.End
		if !finite64(start) || !finite64(end) || start < 0 || end < start || end > e.Duration {
			return fmt.Errorf("word %d has invalid alignment [%g,%g] for duration %g", i, start, end, e.Duration)
		}
		if start < lastStart || end < lastEnd {
			return fmt.Errorf("word %d alignment is not monotonic", i)
		}
		lastStart, lastEnd = start, end
	}
	if len(e.EligibleTrainingCuts(TrainingMinimumCutSeconds)) == 0 {
		return fmt.Errorf("no aligned boundary leaves %.1fs on both sides", TrainingMinimumCutSeconds)
	}
	return nil
}

// ReadAlignedTrainingManifest reads and validates a bounded JSONL manifest.
// maxEntries must be positive so accidental multi-gigabyte admission is
// explicit at the call site.
func ReadAlignedTrainingManifest(r io.Reader, maxEntries int) ([]TrainingEntry, error) {
	if r == nil {
		return nil, fmt.Errorf("Pocket TTS training manifest reader is nil")
	}
	if maxEntries <= 0 {
		return nil, fmt.Errorf("Pocket TTS training manifest max entries must be positive")
	}
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64<<10), trainingManifestLineLimit)
	entries := make([]TrainingEntry, 0, min(maxEntries, 1024))
	line := 0
	for scanner.Scan() {
		line++
		raw := strings.TrimSpace(scanner.Text())
		if raw == "" {
			return nil, fmt.Errorf("Pocket TTS training manifest line %d is blank", line)
		}
		if len(entries) == maxEntries {
			return nil, fmt.Errorf("Pocket TTS training manifest exceeds %d entries", maxEntries)
		}
		var entry TrainingEntry
		if err := json.Unmarshal([]byte(raw), &entry); err != nil {
			return nil, fmt.Errorf("Pocket TTS training manifest line %d: %w", line, err)
		}
		if err := entry.Validate(); err != nil {
			return nil, fmt.Errorf("Pocket TTS training manifest line %d: %w", line, err)
		}
		entries = append(entries, entry)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read Pocket TTS training manifest: %w", err)
	}
	if len(entries) == 0 {
		return nil, fmt.Errorf("Pocket TTS training manifest is empty")
	}
	return entries, nil
}

func LoadAlignedTrainingManifest(path string, maxEntries int) ([]TrainingEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open Pocket TTS training manifest: %w", err)
	}
	defer f.Close()
	return ReadAlignedTrainingManifest(f, maxEntries)
}

// validateLatentEntry preserves upstream latent-mode fallback semantics: rows
// without an eligible aligned cut train from frame zero and the full transcript.
func (e TrainingEntry) validateLatentEntry() error {
	if strings.TrimSpace(e.Path) == "" || !finite64(e.Duration) || e.Duration <= 0 || !finite64(e.Start) || e.Start < 0 || strings.TrimSpace(e.Transcript) == "" || strings.TrimSpace(e.LatentsFile) == "" || len(e.Words) == 0 {
		return fmt.Errorf("invalid latent manifest entry")
	}
	lastStart, lastEnd := -1.0, -1.0
	for i, word := range e.Words {
		if strings.TrimSpace(word.Word) == "" {
			return fmt.Errorf("word %d is empty", i)
		}
		if word.Start != nil {
			start := *word.Start
			if !finite64(start) || start < 0 || start > e.Duration || start < lastStart {
				return fmt.Errorf("word %d has invalid start", i)
			}
			lastStart = start
		}
		if word.End != nil {
			end := *word.End
			if !finite64(end) || end < 0 || end > e.Duration || end < lastEnd || (word.Start != nil && end < *word.Start) {
				return fmt.Errorf("word %d has invalid end", i)
			}
			lastEnd = end
		}
	}
	return nil
}

func loadLatentTrainingManifest(path string, maxEntries int) ([]TrainingEntry, error) {
	if maxEntries <= 0 {
		return nil, fmt.Errorf("Pocket TTS latent manifest max entries must be positive")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open Pocket TTS latent manifest: %w", err)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 64<<10), trainingManifestLineLimit)
	entries := make([]TrainingEntry, 0, min(maxEntries, 1024))
	line := 0
	for scanner.Scan() {
		line++
		raw := strings.TrimSpace(scanner.Text())
		if raw == "" {
			return nil, fmt.Errorf("Pocket TTS latent manifest line %d is blank", line)
		}
		if len(entries) == maxEntries {
			return nil, fmt.Errorf("Pocket TTS latent manifest exceeds %d entries", maxEntries)
		}
		var entry TrainingEntry
		if err = json.Unmarshal([]byte(raw), &entry); err != nil {
			return nil, fmt.Errorf("Pocket TTS latent manifest line %d: %w", line, err)
		}
		if err = entry.validateLatentEntry(); err != nil {
			return nil, fmt.Errorf("Pocket TTS latent manifest line %d: %w", line, err)
		}
		entries = append(entries, entry)
	}
	if err = scanner.Err(); err != nil {
		return nil, fmt.Errorf("read Pocket TTS latent manifest: %w", err)
	}
	if len(entries) == 0 {
		return nil, fmt.Errorf("Pocket TTS latent manifest is empty")
	}
	return entries, nil
}
