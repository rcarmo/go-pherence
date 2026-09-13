package omnivoice

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

const (
	maxChunkPlanRunes  = 16000
	maxChunkPlanChunks = 128

	durationLowThreshold = 50.0
	durationBoostPower   = 3.0
)

var (
	chunkSentencePunctuation = map[rune]bool{
		'.': true, ',': true, ';': true, ':': true, '!': true, '?': true,
		'。': true, '，': true, '；': true, '：': true, '！': true, '？': true, '、': true, '…': true,
	}
	chunkClosingMarks = map[rune]bool{
		'"': true, '\'': true,
		'“': true, '”': true, '‘': true, '’': true,
		')': true, ']': true, '>': true,
		'）': true, '》': true, '」': true, '】': true,
	}
	chunkAbbreviations = map[string]bool{
		"Mr.": true, "Mrs.": true, "Ms.": true, "Dr.": true, "Prof.": true,
		"Sr.": true, "Jr.": true, "Rev.": true, "Fr.": true, "Hon.": true,
		"Pres.": true, "Gov.": true, "Capt.": true, "Gen.": true, "Sen.": true,
		"Rep.": true, "Col.": true, "Maj.": true, "Lt.": true, "Cmdr.": true,
		"Sgt.": true, "Cpl.": true, "Co.": true, "Corp.": true, "Inc.": true,
		"Ltd.": true, "Est.": true, "Dept.": true, "St.": true, "Ave.": true,
		"Blvd.": true, "Rd.": true, "Mt.": true, "Ft.": true, "No.": true,
		"Jan.": true, "Feb.": true, "Mar.": true, "Apr.": true, "Aug.": true,
		"Sep.": true, "Sept.": true, "Oct.": true, "Nov.": true, "Dec.": true,
		"i.e.": true, "e.g.": true, "vs.": true, "Vs.": true, "Etc.": true,
		"approx.": true, "fig.": true, "def.": true,
	}
	chunkUnicodeWeights = []unicodeWeightRange{
		{0x02AF, 1.0}, {0x03FF, 1.0}, {0x052F, 1.0}, {0x058F, 1.0},
		{0x05FF, 1.5}, {0x077F, 1.5}, {0x089F, 1.5}, {0x08FF, 1.5},
		{0x097F, 1.8}, {0x09FF, 1.8}, {0x0A7F, 1.8}, {0x0AFF, 1.8},
		{0x0B7F, 1.8}, {0x0BFF, 1.8}, {0x0C7F, 1.8}, {0x0CFF, 1.8},
		{0x0D7F, 1.8}, {0x0DFF, 1.8}, {0x0EFF, 1.5}, {0x0FFF, 1.8},
		{0x109F, 1.8}, {0x10FF, 1.0}, {0x11FF, 2.5}, {0x137F, 3.0},
		{0x139F, 3.0}, {0x13FF, 1.0}, {0x167F, 1.0}, {0x169F, 1.0},
		{0x16FF, 1.0}, {0x171F, 1.0}, {0x173F, 1.0}, {0x175F, 1.0},
		{0x177F, 1.0}, {0x17FF, 1.8}, {0x18AF, 1.0}, {0x18FF, 1.0},
		{0x194F, 1.8}, {0x19DF, 1.8}, {0x19FF, 1.8}, {0x1A1F, 1.8},
		{0x1AAF, 1.8}, {0x1B7F, 1.8}, {0x1BBF, 1.8}, {0x1BFF, 1.8},
		{0x1C4F, 1.8}, {0x1C7F, 1.8}, {0x1C8F, 1.0}, {0x1CBF, 1.0},
		{0x1CCF, 1.8}, {0x1CFF, 1.8}, {0x1D7F, 1.0}, {0x1DBF, 1.0},
		{0x1DFF, 1.0}, {0x1EFF, 1.0}, {0x309F, 2.2}, {0x30FF, 2.2},
		{0x312F, 3.0}, {0x318F, 2.5}, {0x9FFF, 3.0}, {0xA4CF, 3.0},
		{0xA4FF, 1.0}, {0xA63F, 1.0}, {0xA69F, 1.0}, {0xA6FF, 1.0},
		{0xA7FF, 1.0}, {0xA82F, 1.8}, {0xA87F, 1.0}, {0xA8DF, 1.8},
		{0xA8FF, 1.8}, {0xA92F, 1.8}, {0xA95F, 1.8}, {0xA97F, 2.5},
		{0xA9DF, 1.8}, {0xA9FF, 1.8}, {0xAA5F, 1.8}, {0xAA7F, 1.8},
		{0xAADF, 1.8}, {0xAAFF, 1.8}, {0xAB2F, 3.0}, {0xAB6F, 1.0},
		{0xABBF, 1.0}, {0xABFF, 1.8}, {0xD7AF, 2.5}, {0xFAFF, 3.0},
		{0xFDFF, 1.5}, {0xFE6F, 1.0}, {0xFEFF, 1.5}, {0xFFEF, 1.0},
	}
)

type unicodeWeightRange struct {
	end    rune
	weight float64
}

type chunkPlanner struct {
	cfg       Config
	tok       *tokenizer.Tokenizer
	text      string
	ref       CachedReferenceTokens
	opts      PreparePromptOptions
	maxFrames int
	firstMax  int

	runes         []rune
	offsets       []int
	prefixWeights []float64
	allEnds       []int
	safeEnds      []int
	wordEnds      []int
	sentenceEnds  []int
	refWeight     float64
	refFrames     float64
}

// PlanChunks prepares one or more inference prompts for long target text.
//
// The planner trims only leading and trailing whitespace from text, then splits
// the remaining content into contiguous slices. Concatenating prompt.Text for
// all returned prompts reconstructs that trimmed text exactly.
func PlanChunks(cfg Config, tok *tokenizer.Tokenizer, text string, ref CachedReferenceTokens, opts PreparePromptOptions, maxFrames int) ([]PreparedPrompt, error) {
	return PlanChunksWithFirstLimit(cfg, tok, text, ref, opts, maxFrames, 0)
}

// PlanChunksWithFirstLimit prepares one or more inference prompts for long
// target text, optionally applying a stricter frame budget to only the first
// chunk. When firstFrames is zero, behavior matches PlanChunks exactly.
func PlanChunksWithFirstLimit(cfg Config, tok *tokenizer.Tokenizer, text string, ref CachedReferenceTokens, opts PreparePromptOptions, maxFrames, firstFrames int) ([]PreparedPrompt, error) {
	if !utf8.ValidString(text) {
		return nil, fmt.Errorf("omnivoice: invalid UTF-8 text")
	}
	if tok == nil {
		return nil, fmt.Errorf("omnivoice: nil tokenizer")
	}
	if maxFrames <= 0 || maxFrames > 250 {
		return nil, fmt.Errorf("omnivoice: max_frames must be 1..250")
	}
	if firstFrames != 0 && (firstFrames < 1 || firstFrames > maxFrames) {
		return nil, fmt.Errorf("omnivoice: first_frames must be 1..max_frames")
	}
	if err := ref.Validate(); err != nil {
		return nil, err
	}
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return nil, fmt.Errorf("omnivoice: target text is required")
	}
	if utf8.RuneCountInString(trimmed) > maxChunkPlanRunes {
		return nil, fmt.Errorf("omnivoice: target text exceeds %d runes", maxChunkPlanRunes)
	}
	planner := newChunkPlanner(cfg, tok, trimmed, ref, opts, maxFrames, firstFrames)
	return planner.plan()
}

func newChunkPlanner(cfg Config, tok *tokenizer.Tokenizer, text string, ref CachedReferenceTokens, opts PreparePromptOptions, maxFrames, firstFrames int) *chunkPlanner {
	runes := []rune(text)
	offsets := make([]int, 0, len(runes)+1)
	for at := range text {
		offsets = append(offsets, at)
	}
	offsets = append(offsets, len(text))

	prefixWeights := make([]float64, len(runes)+1)
	for i, r := range runes {
		prefixWeights[i+1] = prefixWeights[i] + chunkCharWeight(r)
	}

	disallow := buildDisallowedBoundaries(runes)
	allEnds, safeEnds, wordEnds, sentenceEnds := buildChunkBoundaries(runes, disallow)

	refWeight := chunkTotalWeight(ref.Transcript)
	refFrames := float64(ref.Frames)
	if refWeight <= 0 || refFrames <= 0 {
		refWeight = chunkTotalWeight("Nice to meet you.")
		refFrames = 25
	}

	return &chunkPlanner{
		cfg:           cfg,
		tok:           tok,
		text:          text,
		ref:           ref,
		opts:          opts,
		maxFrames:     maxFrames,
		firstMax:      firstFrames,
		runes:         runes,
		offsets:       offsets,
		prefixWeights: prefixWeights,
		allEnds:       allEnds,
		safeEnds:      safeEnds,
		wordEnds:      wordEnds,
		sentenceEnds:  sentenceEnds,
		refWeight:     refWeight,
		refFrames:     refFrames,
	}
}

func (p *chunkPlanner) plan() ([]PreparedPrompt, error) {
	wholeLimit := p.frameLimit(0)
	wholeFrames := p.targetFrames(0, len(p.runes), wholeLimit)
	if p.estimateFrames(0, len(p.runes)) <= wholeLimit {
		prompt, err := p.prepare(0, len(p.runes), wholeFrames)
		if err == nil {
			return []PreparedPrompt{prompt}, nil
		}
	}

	chunks := make([]PreparedPrompt, 0, 4)
	for start := 0; start < len(p.runes); {
		if len(chunks) >= maxChunkPlanChunks {
			return nil, fmt.Errorf("omnivoice: chunk planning exceeded %d chunks", maxChunkPlanChunks)
		}
		end, prompt, err := p.planOne(start, p.frameLimit(start))
		if err != nil {
			return nil, err
		}
		chunks = append(chunks, prompt)
		start = end
	}
	return chunks, nil
}

func (p *chunkPlanner) frameLimit(start int) int {
	if start == 0 && p.firstMax > 0 {
		return p.firstMax
	}
	return p.maxFrames
}

func (p *chunkPlanner) planOne(start, frameLimit int) (int, PreparedPrompt, error) {
	limit := p.maxEndWithinFrameBudget(start, frameLimit)
	if limit <= start {
		limit = p.nextSafeEnd(start)
		if limit <= start {
			limit = start + 1
		}
	}

	for _, ends := range [][]int{
		p.boundaryWindow(p.sentenceEnds, start, limit),
		p.boundaryWindow(p.wordEnds, start, limit),
		p.boundaryWindow(p.safeEnds, start, limit),
		p.boundaryWindow(p.allEnds, start, limit),
	} {
		if end, prompt, ok := p.searchBestFit(start, ends, frameLimit); ok {
			return end, prompt, nil
		}
	}

	for end := start + 1; end <= minInt(len(p.runes), maxInt(limit, start+1)); end++ {
		prompt, err := p.prepare(start, end, p.targetFrames(start, end, frameLimit))
		if err == nil {
			return end, prompt, nil
		}
	}

	return 0, PreparedPrompt{}, fmt.Errorf("omnivoice: no prompt capacity for chunk starting at rune %d (reference frames=%d, max_frames=%d)", start, p.ref.Frames, frameLimit)
}

func (p *chunkPlanner) searchBestFit(start int, ends []int, frameLimit int) (int, PreparedPrompt, bool) {
	if len(ends) == 0 {
		return 0, PreparedPrompt{}, false
	}
	lo, hi := 0, len(ends)-1
	bestIdx := -1
	var bestPrompt PreparedPrompt
	for lo <= hi {
		mid := (lo + hi) / 2
		end := ends[mid]
		prompt, err := p.prepare(start, end, p.targetFrames(start, end, frameLimit))
		if err == nil {
			bestIdx = mid
			bestPrompt = prompt
			lo = mid + 1
			continue
		}
		hi = mid - 1
	}
	if bestIdx < 0 {
		return 0, PreparedPrompt{}, false
	}
	for i := bestIdx + 1; i < len(ends) && i <= bestIdx+4; i++ {
		prompt, err := p.prepare(start, ends[i], p.targetFrames(start, ends[i], frameLimit))
		if err == nil {
			bestIdx = i
			bestPrompt = prompt
		}
	}
	return ends[bestIdx], bestPrompt, true
}

func (p *chunkPlanner) prepare(start, end, frames int) (PreparedPrompt, error) {
	text := p.text[p.offsets[start]:p.offsets[end]]
	return PrepareInferenceInputs(p.cfg, p.tok, text, frames, p.ref, p.opts)
}

func (p *chunkPlanner) maxEndWithinFrameBudget(start, frameLimit int) int {
	lo, hi := start+1, len(p.runes)
	best := start
	for lo <= hi {
		mid := (lo + hi) / 2
		if p.estimateFrames(start, mid) <= frameLimit {
			best = mid
			lo = mid + 1
		} else {
			hi = mid - 1
		}
	}
	return best
}

func (p *chunkPlanner) nextSafeEnd(start int) int {
	at := sort.SearchInts(p.safeEnds, start+1)
	if at < len(p.safeEnds) {
		return p.safeEnds[at]
	}
	if start < len(p.runes) {
		return start + 1
	}
	return start
}

func (p *chunkPlanner) boundaryWindow(ends []int, start, limit int) []int {
	if len(ends) == 0 || limit <= start {
		return nil
	}
	lo := sort.SearchInts(ends, start+1)
	hi := sort.Search(len(ends), func(i int) bool { return ends[i] > limit })
	if lo >= hi {
		return nil
	}
	return ends[lo:hi]
}

func (p *chunkPlanner) targetFrames(start, end, frameLimit int) int {
	frames := p.estimateFrames(start, end)
	if frames < 1 {
		frames = 1
	}
	if frames > frameLimit {
		frames = frameLimit
	}
	return frames
}

func (p *chunkPlanner) estimateFrames(start, end int) int {
	weight := p.prefixWeights[end] - p.prefixWeights[start]
	est := estimateWeightedDuration(weight, p.refWeight, p.refFrames)
	if est <= 0 {
		return 1
	}
	return maxInt(1, int(est))
}

func estimateWeightedDuration(targetWeight, refWeight, refFrames float64) float64 {
	if targetWeight <= 0 || refWeight <= 0 || refFrames <= 0 {
		return 0
	}
	speedFactor := refWeight / refFrames
	raw := targetWeight / speedFactor
	if raw < durationLowThreshold {
		alpha := 1.0 / durationBoostPower
		return durationLowThreshold * math.Pow(raw/durationLowThreshold, alpha)
	}
	return raw
}

func buildDisallowedBoundaries(runes []rune) []bool {
	disallow := make([]bool, len(runes)+1)
	for i := 0; i < len(runes); i++ {
		if runes[i] == '<' && i+1 < len(runes) && runes[i+1] == '|' {
			end := -1
			for j := i + 2; j+1 < len(runes); j++ {
				if runes[j] == '|' && runes[j+1] == '>' {
					end = j + 1
					break
				}
			}
			if end >= 0 {
				for boundary := i + 1; boundary <= end; boundary++ {
					disallow[boundary] = true
				}
				i = end
			}
			continue
		}
		if runes[i] != '[' {
			continue
		}
		end := -1
		for j := i + 1; j < len(runes); j++ {
			if runes[j] == '[' {
				end = -1
				break
			}
			if runes[j] == ']' {
				end = j
				break
			}
		}
		if end < 0 {
			continue
		}
		for pos := i + 1; pos <= end; pos++ {
			disallow[pos] = true
		}
		i = end
	}
	return disallow
}

func buildChunkBoundaries(runes []rune, disallow []bool) (allEnds, safeEnds, wordEnds, sentenceEnds []int) {
	allEnds = make([]int, 0, len(runes))
	safeEnds = make([]int, 0, len(runes))
	wordEnds = make([]int, 0, len(runes)/2)
	sentenceEnds = make([]int, 0, len(runes)/4)

	for end := 1; end <= len(runes); end++ {
		if !disallow[end] {
			allEnds = append(allEnds, end)
			safeEnds = append(safeEnds, end)
		}
	}

	for i := 0; i < len(runes); i++ {
		if !chunkSentencePunctuation[runes[i]] {
			continue
		}
		if runes[i] == '.' && chunkDotAbbreviation(runes, i) {
			continue
		}
		end := i + 1
		for end < len(runes) && chunkClosingMarks[runes[end]] {
			end++
		}
		for end < len(runes) && unicode.IsSpace(runes[end]) {
			end++
		}
		if end <= len(runes) && !disallow[end] {
			sentenceEnds = appendUniqueInt(sentenceEnds, end)
			wordEnds = appendUniqueInt(wordEnds, end)
		}
	}

	for i := 0; i < len(runes); {
		if !unicode.IsSpace(runes[i]) {
			i++
			continue
		}
		end := i + 1
		for end < len(runes) && unicode.IsSpace(runes[end]) {
			end++
		}
		if !disallow[end] {
			wordEnds = appendUniqueInt(wordEnds, end)
		}
		i = end
	}

	return allEnds, safeEnds, wordEnds, sentenceEnds
}

func chunkDotAbbreviation(runes []rune, dot int) bool {
	start := dot
	for start > 0 && !unicode.IsSpace(runes[start-1]) {
		start--
	}
	if start >= dot {
		return false
	}
	_, hadSpace := trimSpaceRunes(runes[start : dot+1])
	if hadSpace {
		return false
	}
	return chunkAbbreviations[string(runes[start:dot+1])]
}

func trimSpaceRunes(runes []rune) ([]rune, bool) {
	start, end := 0, len(runes)
	for start < end && unicode.IsSpace(runes[start]) {
		start++
	}
	for end > start && unicode.IsSpace(runes[end-1]) {
		end--
	}
	return runes[start:end], start > 0 || end < len(runes)
}

func appendUniqueInt(dst []int, v int) []int {
	if len(dst) != 0 && dst[len(dst)-1] == v {
		return dst
	}
	return append(dst, v)
}

func chunkTotalWeight(text string) float64 {
	var total float64
	for _, r := range text {
		total += chunkCharWeight(r)
	}
	return total
}

func chunkCharWeight(r rune) float64 {
	code := int(r)
	if ('A' <= r && r <= 'Z') || ('a' <= r && r <= 'z') {
		return 1.0
	}
	if r == ' ' {
		return 0.2
	}
	if code == 0x0640 {
		return 0.0
	}
	if unicode.IsMark(r) {
		return 0.0
	}
	if unicode.IsPunct(r) || unicode.IsSymbol(r) {
		return 0.5
	}
	if unicode.IsSpace(r) {
		return 0.2
	}
	if unicode.IsNumber(r) {
		return 3.5
	}
	idx := sort.Search(len(chunkUnicodeWeights), func(i int) bool {
		return r <= chunkUnicodeWeights[i].end
	})
	if idx < len(chunkUnicodeWeights) {
		return chunkUnicodeWeights[idx].weight
	}
	if code > 0x20000 {
		return 3.0
	}
	return 1.0
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
