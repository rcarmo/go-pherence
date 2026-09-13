package whisper

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"unicode"
)

var (
	ErrWordAlignmentConfig = errors.New("invalid Whisper word-alignment configuration")
	ErrWordAlignmentInput  = errors.New("invalid Whisper word-alignment input")
)

// WordTiming is an owned word-level alignment in seconds relative to the audio
// represented by the DecoderState cross-attention cache. TokenStart/TokenEnd
// select the original text-token slice; Speaker remains deliberately absent.
type WordTiming struct {
	Word                 string
	Start, End           float64
	TokenStart, TokenEnd int
}

// AlignWordsChecked teacher-forces text tokens through a fresh DecoderState,
// observes only the alignment heads pinned by CheckedGenerationConfig, applies
// the Transformers 4.57.1 normalise/median/mean/DTW contract, and groups token
// times into whitespace-delimited words. It is CPU-only and mutates state by
// consuming the prompt+tokens. Construct a dedicated state from the matching
// encoder output; do not reuse generation state or call concurrently. audioFrames
// is the unpadded 10 ms log-mel frame count, matching Transformers num_frames.
func AlignWordsChecked(ctx context.Context, dec *Decoder, state *DecoderState, tokenizer *Tokenizer, generation *CheckedGenerationConfig, language string, tokens []int, audioFrames int) ([]WordTiming, error) {
	if ctx == nil {
		return nil, fmt.Errorf("%w: nil context", ErrWordAlignmentInput)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if dec == nil || state == nil || tokenizer == nil || generation == nil {
		return nil, fmt.Errorf("%w: nil model/state/tokenizer/generation", ErrWordAlignmentInput)
	}
	if generation.cfg != dec.cfg || state.Pos != 0 || state.LastToken != -1 || len(tokens) == 0 || audioFrames <= 0 {
		return nil, fmt.Errorf("%w: model/state/token/frame contract", ErrWordAlignmentInput)
	}
	if state.CrossAttentionObserver != nil || len(state.CrossK) != dec.cfg.DecoderLayers || len(state.CrossKHead) != dec.cfg.DecoderLayers || len(generation.alignmentHeads) == 0 {
		return nil, fmt.Errorf("%w: observer/cross-K/alignment-head contract", ErrWordAlignmentConfig)
	}
	languageToken, ok := generation.languages[language]
	if !ok {
		return nil, ErrWordAlignmentConfig
	}
	for _, token := range tokens {
		if token < 0 || token >= generation.vocabulary.eot {
			return nil, fmt.Errorf("%w: text token %d", ErrWordAlignmentInput, token)
		}
	}
	encLen := len(state.CrossK[0]) / dec.cfg.DecoderDModel
	if encLen < 1 || encLen > (dec.cfg.MaxLength+1)/2 {
		return nil, fmt.Errorf("%w: encoder length %d", ErrWordAlignmentInput, encLen)
	}
	alignmentFrames := min(encLen, audioFrames/2)
	if alignmentFrames < 1 {
		return nil, fmt.Errorf("%w: alignment frame extent", ErrWordAlignmentInput)
	}
	for layer := range state.CrossK {
		if len(state.CrossK[layer]) != encLen*dec.cfg.DecoderDModel || len(state.CrossKHead[layer]) != len(state.CrossK[layer]) {
			return nil, fmt.Errorf("%w: cross-K layer %d shape", ErrWordAlignmentInput, layer)
		}
	}
	if len(tokens)+4 > dec.cfg.MaxDecoderLength {
		return nil, fmt.Errorf("%w: decoder sequence length", ErrWordAlignmentInput)
	}

	headIndex := make(map[[2]int]int, len(generation.alignmentHeads))
	for index, head := range generation.alignmentHeads {
		headIndex[head] = index
	}
	weights := make([]float32, len(generation.alignmentHeads)*len(tokens)*alignmentFrames)
	observed := make([]bool, len(generation.alignmentHeads)*len(tokens))
	position := -1
	state.CrossAttentionObserver = func(layer, head, tokenPosition int, row []float32) {
		index, selected := headIndex[[2]int{layer, head}]
		if !selected || position < 0 || position >= len(tokens) || tokenPosition != position+4 || len(row) != encLen {
			return
		}
		copy(weights[(index*len(tokens)+position)*alignmentFrames:], row[:alignmentFrames])
		observed[index*len(tokens)+position] = true
	}
	defer func() { state.CrossAttentionObserver = nil }()
	var logits []float32
	for index, token := range []int{generation.vocabulary.sot, languageToken, generation.vocabulary.transcribe, generation.vocabulary.noTimestamps} {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		logits = dec.ForwardToken(token, state)
		if len(logits) != dec.cfg.VocabSize {
			return nil, fmt.Errorf("%w: prefix logits at position %d", ErrWordAlignmentInput, index)
		}
	}
	for index, token := range tokens {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		position = index
		logits = dec.ForwardToken(token, state)
		if len(logits) != dec.cfg.VocabSize {
			return nil, fmt.Errorf("%w: text logits at position %d", ErrWordAlignmentInput, index)
		}
	}
	position = -1
	if slices.Contains(observed, false) {
		return nil, fmt.Errorf("%w: incomplete cross-attention observation", ErrWordAlignmentInput)
	}

	matrix, err := alignmentCost(ctx, weights, len(generation.alignmentHeads), len(tokens), alignmentFrames, 7)
	if err != nil {
		return nil, fmt.Errorf("alignment cost: %w", err)
	}
	text, times, err := dynamicTimeWarp(ctx, matrix, len(tokens), alignmentFrames)
	if err != nil {
		return nil, fmt.Errorf("DTW: %w", err)
	}
	starts := make([]int, len(tokens)+1)
	jump := true
	lastText := -1
	at := 0
	for i := range text {
		if i > 0 {
			jump = text[i] != text[i-1]
		}
		if jump && text[i] >= 0 && text[i] < len(starts) {
			if at != text[i] {
				return nil, fmt.Errorf("%w: discontinuous DTW token timestamps", ErrWordAlignmentInput)
			}
			starts[at] = times[i]
			at++
		}
		lastText = text[i]
	}
	if lastText != len(tokens)-1 {
		return nil, fmt.Errorf("%w: DTW terminal text index %d", ErrWordAlignmentInput, lastText)
	}
	if at != len(tokens) {
		return nil, fmt.Errorf("%w: incomplete DTW token timestamps", ErrWordAlignmentInput)
	}
	starts[len(tokens)] = starts[len(tokens)-1]
	for i := 1; i < len(starts); i++ {
		if starts[i] < starts[i-1] {
			return nil, fmt.Errorf("%w: decreasing token timestamps", ErrWordAlignmentInput)
		}
	}
	groups, err := tokenizer.wordGroups(tokens)
	if err != nil {
		return nil, fmt.Errorf("token-to-word grouping: %w", err)
	}
	out := make([]WordTiming, 0, len(groups))
	for _, group := range groups {
		start := .02 * float64(starts[group.start])
		end := .02 * float64(starts[group.end])
		if end < start || start < 0 || starts[group.end] > alignmentFrames {
			return nil, fmt.Errorf("%w: word timestamp bounds", ErrWordAlignmentInput)
		}
		out = append(out, WordTiming{Word: group.word, Start: start, End: end, TokenStart: group.start, TokenEnd: group.end})
	}
	return out, ctx.Err()
}

func alignmentCost(ctx context.Context, weights []float32, heads, text, frames, width int) ([]float64, error) {
	if heads < 1 || text < 1 || frames < 1 || width < 1 || width%2 == 0 || len(weights) != heads*text*frames {
		return nil, fmt.Errorf("%w: attention matrix shape", ErrWordAlignmentInput)
	}
	filtered := make([]float64, len(weights))
	half := width / 2
	window := make([]float32, width)
	for head := 0; head < heads; head++ {
		for frame := 0; frame < frames; frame++ {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			var mean float32
			for token := 0; token < text; token++ {
				mean += weights[(head*text+token)*frames+frame]
			}
			mean /= float32(text)
			var variance float32
			for token := 0; token < text; token++ {
				delta := weights[(head*text+token)*frames+frame] - mean
				variance += delta * delta
			}
			std := float32(math.Sqrt(float64(variance / float32(text))))
			if std == 0 || math.IsNaN(float64(std)) || math.IsInf(float64(std), 0) {
				return nil, fmt.Errorf("%w: zero/non-finite attention variance in head %d frame %d", ErrWordAlignmentInput, head, frame)
			}
			for token := 0; token < text; token++ {
				weights[(head*text+token)*frames+frame] = (weights[(head*text+token)*frames+frame] - mean) / std
			}
		}
		for token := 0; token < text; token++ {
			row := weights[(head*text+token)*frames : (head*text+token+1)*frames]
			for frame := 0; frame < frames; frame++ {
				if frames <= half {
					filtered[(head*text+token)*frames+frame] = float64(row[frame])
					continue
				}
				for k := -half; k <= half; k++ {
					index := frame + k
					if index < 0 {
						index = -index
					} else if index >= frames {
						index = 2*frames - index - 2
					}
					window[k+half] = row[index]
				}
				slices.Sort(window)
				filtered[(head*text+token)*frames+frame] = float64(window[half])
			}
		}
	}
	matrix := make([]float64, text*frames)
	for token := 0; token < text; token++ {
		for frame := 0; frame < frames; frame++ {
			var sum float64
			for head := 0; head < heads; head++ {
				sum += filtered[(head*text+token)*frames+frame]
			}
			matrix[token*frames+frame] = -float64(float32(sum) / float32(heads))
		}
	}
	return matrix, nil
}

func dynamicTimeWarp(ctx context.Context, matrix []float64, rows, cols int) ([]int, []int, error) {
	if rows < 1 || cols < 1 || len(matrix) != rows*cols || int64(rows+1)*int64(cols+1) > 1<<24 {
		return nil, nil, fmt.Errorf("%w: DTW matrix shape", ErrWordAlignmentInput)
	}
	stride := cols + 1
	cost := make([]float32, (rows+1)*stride)
	trace := make([]uint8, len(cost))
	for i := range cost {
		cost[i] = float32(math.Inf(1))
		trace[i] = 255
	}
	cost[0] = 0
	for col := 1; col <= cols; col++ {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		for row := 1; row <= rows; row++ {
			c0 := cost[(row-1)*stride+col-1]
			c1 := cost[(row-1)*stride+col]
			c2 := cost[row*stride+col-1]
			var c float32
			var direction uint8
			if c0 < c1 && c0 < c2 {
				c, direction = c0, 0
			} else if c1 < c0 && c1 < c2 {
				c, direction = c1, 1
			} else {
				c, direction = c2, 2
			}
			value := matrix[(row-1)*cols+col-1]
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, nil, fmt.Errorf("%w: non-finite DTW cost", ErrWordAlignmentInput)
			}
			cost[row*stride+col] = float32(value) + c
			trace[row*stride+col] = direction
		}
	}
	for col := 0; col <= cols; col++ {
		trace[col] = 2
	}
	for row := 0; row <= rows; row++ {
		trace[row*stride] = 1
	}
	row, col := rows, cols
	text, times := make([]int, 0, rows+cols), make([]int, 0, rows+cols)
	for row > 0 || col > 0 {
		text = append(text, row-1)
		times = append(times, col-1)
		switch trace[row*stride+col] {
		case 0:
			row--
			col--
		case 1:
			row--
		case 2:
			col--
		default:
			return nil, nil, ErrWordAlignmentInput
		}
	}
	slices.Reverse(text)
	slices.Reverse(times)
	return text, times, nil
}

type wordGroup struct {
	word       string
	start, end int
}

func (t *Tokenizer) wordGroups(tokens []int) ([]wordGroup, error) {
	if t == nil || len(tokens) == 0 {
		return nil, ErrWordAlignmentInput
	}
	var groups []wordGroup
	start := 0
	previous := ""
	for i := range tokens {
		if tokens[i] < 0 || tokens[i] >= TokenSOT {
			return nil, ErrWordAlignmentInput
		}
		piece := t.decodeRaw(tokens[i : i+1])
		decoded := t.decodeRaw(tokens[start : i+1])
		if decoded == "" {
			continue
		}
		spaceBoundary := i > start && strings.HasPrefix(piece, " ")
		punctuation := allPunctuation([]rune(strings.TrimSpace(piece)))
		if spaceBoundary {
			if previous == "" {
				return nil, ErrWordAlignmentInput
			}
			groups = append(groups, wordGroup{word: strings.TrimSpace(previous), start: start, end: i})
			start = i
			decoded = t.decodeRaw(tokens[start : i+1])
		} else if punctuation {
			if previous == "" {
				if len(groups) == 0 {
					return nil, ErrWordAlignmentInput
				}
				groups[len(groups)-1].word += strings.TrimSpace(piece)
				groups[len(groups)-1].end = i + 1
			} else {
				groups = append(groups, wordGroup{word: strings.TrimSpace(previous) + strings.TrimSpace(piece), start: start, end: i + 1})
			}
			start = i + 1
			previous = ""
			continue
		}
		previous = decoded
	}
	if previous != "" {
		groups = append(groups, wordGroup{word: strings.TrimSpace(previous), start: start, end: len(tokens)})
	}
	if len(groups) == 0 || groups[len(groups)-1].end != len(tokens) {
		return nil, ErrWordAlignmentInput
	}
	return groups, nil
}

func allPunctuation(value []rune) bool {
	if len(value) == 0 {
		return false
	}
	for _, r := range value {
		if !unicode.IsPunct(r) && !unicode.IsSymbol(r) {
			return false
		}
	}
	return true
}
