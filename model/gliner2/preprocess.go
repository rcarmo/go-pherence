package gliner2

import (
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"

	"golang.org/x/text/cases"
	"golang.org/x/text/language"
)

// Word retains offsets into the original text, never its lowercased form.
// Byte offsets slice Go strings; character offsets match Python's output API.
type Word struct {
	Text                           string
	ByteStart, ByteEnd, Start, End int
}

var wordPattern = regexp.MustCompile(`(?i)https?://[^\s]+|www\.[^\s]+|[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}|@[a-z0-9_]+|[\p{L}\p{N}_]+(?:[-_][\p{L}\p{N}_]+)*|[^\s]`)

func SplitWords(text string) ([]Word, error) {
	if !utf8.ValidString(text) {
		return nil, fmt.Errorf("invalid UTF-8")
	}
	lower := cases.Lower(language.Und)
	var words []Word
	for _, loc := range wordPattern.FindAllStringIndex(text, -1) {
		token := text[loc[0]:loc[1]]
		// RE2's \s is ASCII; Python's \s includes Unicode whitespace.
		if strings.TrimSpace(token) == "" {
			continue
		}
		words = append(words, Word{Text: lower.String(token), ByteStart: loc[0], ByteEnd: loc[1], Start: utf8.RuneCountInString(text[:loc[0]]), End: utf8.RuneCountInString(text[:loc[1]])})
	}
	return words, nil
}

type EntityInput struct {
	IDs            []int
	Words          []Word
	TextPositions  []int
	QueryPositions []int
	Labels         []string
}

// PrepareEntities builds the simple ordered entity schema with first-subword
// routing. Unlike Encode(addSpecial=true), upstream processor formatting does
// NOT wrap this combined sequence in CLS/SEP. Descriptions and other task
// schemas are separate APIs, not silently folded into entity labels.
func (t *Tokenizer) PrepareEntities(text string, labels []string, maxTokens int) (EntityInput, error) {
	return t.prepareSchema(text, "entities", "[E]", labels, maxTokens)
}

// PrepareClassification routes choice markers for a single named task.
func (t *Tokenizer) PrepareClassification(text, task string, labels []string, maxTokens int) (EntityInput, error) {
	if strings.TrimSpace(task) == "" {
		return EntityInput{}, fmt.Errorf("classification task required")
	}
	return t.prepareSchema(text, task, "[L]", labels, maxTokens)
}

func (t *Tokenizer) prepareSchema(text, parent, marker string, labels []string, maxTokens int) (EntityInput, error) {
	if len(labels) == 0 || maxTokens <= 0 {
		return EntityInput{}, fmt.Errorf("entity labels and token budget required")
	}
	words, err := SplitWords(text)
	if err != nil {
		return EntityInput{}, err
	}
	result := EntityInput{Words: words, Labels: append([]string(nil), labels...)}
	appendToken := func(token string) error {
		ids, err := t.Encode(token, false)
		if err != nil {
			return err
		}
		if len(ids) == 0 {
			return fmt.Errorf("schema/text item produced no tokens")
		}
		result.IDs = append(result.IDs, ids...)
		return nil
	}
	for _, s := range []string{"(", "[P]", parent, "("} {
		if err := appendToken(s); err != nil {
			return EntityInput{}, err
		}
	}
	seen := make(map[string]bool)
	for _, label := range labels {
		if strings.TrimSpace(label) == "" || seen[label] {
			return EntityInput{}, fmt.Errorf("empty or duplicate label %q", label)
		}
		seen[label] = true
		result.QueryPositions = append(result.QueryPositions, len(result.IDs))
		if err := appendToken(marker); err != nil {
			return EntityInput{}, err
		}
		if err := appendToken(label); err != nil {
			return EntityInput{}, err
		}
	}
	for _, s := range []string{")", ")", "[SEP_TEXT]"} {
		if err := appendToken(s); err != nil {
			return EntityInput{}, err
		}
	}
	for _, w := range words {
		result.TextPositions = append(result.TextPositions, len(result.IDs))
		if err := appendToken(w.Text); err != nil {
			return EntityInput{}, err
		}
	}
	if len(result.IDs) > maxTokens {
		return EntityInput{}, fmt.Errorf("formatted sequence %d exceeds token budget %d", len(result.IDs), maxTokens)
	}
	return result, nil
}
