package omnivoice

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"regexp"
	"strings"
	"unicode"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

type CachedReferenceTokens struct {
	Books      int      `json:"books"`
	Frames     int      `json:"frames"`
	Codes      []int    `json:"codes"`
	Transcript string   `json:"transcript"`
	RefRMS     *float64 `json:"ref_rms,omitempty"`
	RMS        *float64 `json:"rms,omitempty"`
}

type PreparePromptOptions struct {
	Language string
	Instruct string
	Denoise  bool
}

type PreparedInput struct {
	Tokens    int    `json:"tokens"`
	IDs       []int  `json:"ids"`
	AudioMask []bool `json:"audio_mask"`
}

type PreparedPrompt struct {
	Conditional   PreparedInput `json:"conditional"`
	Unconditional PreparedInput `json:"unconditional"`
	TargetFrames  int           `json:"target_frames"`
	Text          string        `json:"text"`
	RefRMS        *float64      `json:"ref_rms,omitempty"`
}

func LoadCachedReferenceTokens(path string) (CachedReferenceTokens, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return CachedReferenceTokens{}, err
	}
	var ref CachedReferenceTokens
	if err := json.Unmarshal(data, &ref); err != nil {
		return CachedReferenceTokens{}, err
	}
	if err := ref.Validate(); err != nil {
		return CachedReferenceTokens{}, err
	}
	return ref, nil
}

func (r CachedReferenceTokens) Validate() error {
	if r.Books <= 0 || r.Books > 32 {
		return fmt.Errorf("omnivoice: books must be > 0")
	}
	if r.Frames <= 0 || r.Frames > 500 {
		return fmt.Errorf("omnivoice: frames must be > 0")
	}
	if len(r.Codes) != r.Books*r.Frames {
		return fmt.Errorf("omnivoice: codes=%d want %d", len(r.Codes), r.Books*r.Frames)
	}
	for _, rms := range []*float64{r.RefRMS, r.RMS} {
		if rms != nil && (math.IsNaN(*rms) || math.IsInf(*rms, 0) || *rms < 0) {
			return fmt.Errorf("omnivoice: reference RMS must be finite and nonnegative")
		}
	}
	if strings.TrimSpace(r.Transcript) == "" {
		return fmt.Errorf("omnivoice: transcript is required")
	}
	return nil
}

func (r CachedReferenceTokens) EffectiveRMS() *float64 {
	if r.RefRMS != nil {
		return r.RefRMS
	}
	return r.RMS
}

// PrepareInferenceInputs mirrors upstream OmniVoice._prepare_inference_inputs
// for text tokenization and prompt layout, using cached reference audio tokens
// instead of a live reference encoder.
func PrepareInferenceInputs(cfg Config, tok *tokenizer.Tokenizer, text string, targetFrames int, ref CachedReferenceTokens, opts PreparePromptOptions) (PreparedPrompt, error) {
	if tok == nil {
		return PreparedPrompt{}, fmt.Errorf("omnivoice: nil tokenizer")
	}
	if targetFrames <= 0 || targetFrames > 250 {
		return PreparedPrompt{}, fmt.Errorf("omnivoice: target_frames must be 1..250")
	}
	if strings.TrimSpace(text) == "" {
		return PreparedPrompt{}, fmt.Errorf("omnivoice: target text is required")
	}
	if err := ref.Validate(); err != nil {
		return PreparedPrompt{}, err
	}
	if cfg.NumAudioCodebook <= 0 {
		return PreparedPrompt{}, fmt.Errorf("omnivoice: invalid num_audio_codebook")
	}
	if ref.Books != cfg.NumAudioCodebook {
		return PreparedPrompt{}, fmt.Errorf("omnivoice: reference books=%d want %d", ref.Books, cfg.NumAudioCodebook)
	}

	for _, code := range ref.Codes {
		if code < 0 || code >= cfg.AudioVocabSize || code == cfg.AudioMaskID {
			return PreparedPrompt{}, fmt.Errorf("omnivoice: invalid reference audio code %d", code)
		}
	}

	styleText := ""
	if opts.Denoise {
		styleText += "<|denoise|>"
	}
	lang := opts.Language
	if lang == "" {
		lang = "None"
	}
	instruct := opts.Instruct
	if instruct == "" {
		instruct = "None"
	}
	styleText += "<|lang_start|>" + lang + "<|lang_end|>"
	styleText += "<|instruct_start|>" + instruct + "<|instruct_end|>"
	styleIDs := tok.Encode(styleText)

	fullText := combineText(text, ref.Transcript)
	textIDs := tokenizeWithNonverbalTags("<|text_start|>"+fullText+"<|text_end|>", tok)

	rowTokens := len(styleIDs) + len(textIDs) + ref.Frames + targetFrames
	if rowTokens > 512 {
		return PreparedPrompt{}, fmt.Errorf("omnivoice: prepared prompt exceeds 512 positions")
	}
	out := PreparedPrompt{
		TargetFrames: targetFrames,
		Text:         text,
		RefRMS:       ref.EffectiveRMS(),
		Conditional: PreparedInput{
			Tokens:    rowTokens,
			IDs:       make([]int, cfg.NumAudioCodebook*rowTokens),
			AudioMask: make([]bool, rowTokens),
		},
		Unconditional: PreparedInput{
			Tokens:    targetFrames,
			IDs:       make([]int, cfg.NumAudioCodebook*targetFrames),
			AudioMask: make([]bool, targetFrames),
		},
	}

	audioStart := rowTokens - targetFrames - ref.Frames
	for i := audioStart; i < rowTokens; i++ {
		out.Conditional.AudioMask[i] = true
	}
	for i := range out.Unconditional.AudioMask {
		out.Unconditional.AudioMask[i] = true
	}
	for book := 0; book < cfg.NumAudioCodebook; book++ {
		row := out.Conditional.IDs[book*rowTokens : (book+1)*rowTokens]
		at := 0
		at += copy(row[at:], styleIDs)
		at += copy(row[at:], textIDs)
		copy(row[at:at+ref.Frames], ref.Codes[book*ref.Frames:(book+1)*ref.Frames])
		at += ref.Frames
		for i := at; i < rowTokens; i++ {
			row[i] = cfg.AudioMaskID
		}
		for i := 0; i < targetFrames; i++ {
			out.Unconditional.IDs[book*targetFrames+i] = cfg.AudioMaskID
		}
	}
	return out, nil
}

var nonverbalTagPattern = regexp.MustCompile(`\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|surprise-yo|dissatisfaction-hnn)\]`)
var collapseSpacesPattern = regexp.MustCompile(`[ \t]+`)
var stripNewlinesPattern = regexp.MustCompile(`[\r\n]+`)

func tokenizeWithNonverbalTags(text string, tok *tokenizer.Tokenizer) []int {
	parts := make([]int, 0)
	lastEnd := 0
	for _, loc := range nonverbalTagPattern.FindAllStringIndex(text, -1) {
		if loc[0] > lastEnd {
			parts = append(parts, tok.Encode(text[lastEnd:loc[0]])...)
		}
		parts = append(parts, tok.Encode(text[loc[0]:loc[1]])...)
		lastEnd = loc[1]
	}
	if lastEnd < len(text) {
		parts = append(parts, tok.Encode(text[lastEnd:])...)
	}
	if len(parts) == 0 {
		return tok.Encode(text)
	}
	return parts
}

func combineText(text, refText string) string {
	var fullText string
	if strings.TrimSpace(refText) != "" {
		fullText = strings.TrimSpace(refText) + " " + strings.TrimSpace(text)
	} else {
		fullText = strings.TrimSpace(text)
	}
	fullText = stripNewlinesPattern.ReplaceAllString(fullText, "")
	fullText = strings.ReplaceAll(fullText, "（", "(")
	fullText = strings.ReplaceAll(fullText, "）", ")")
	fullText = collapseSpacesPattern.ReplaceAllString(fullText, " ")
	return removeSpacesAroundChinese(fullText)
}

func removeSpacesAroundChinese(text string) string {
	runes := []rune(text)
	out := make([]rune, 0, len(runes))
	for i, r := range runes {
		if !unicode.IsSpace(r) {
			out = append(out, r)
			continue
		}
		prevChinese := i > 0 && isChineseRune(runes[i-1])
		nextChinese := i+1 < len(runes) && isChineseRune(runes[i+1])
		if prevChinese || nextChinese {
			continue
		}
		out = append(out, r)
	}
	return string(out)
}

func isChineseRune(r rune) bool {
	return r >= 0x4e00 && r <= 0x9fff
}
