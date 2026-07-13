package mosstranscribe

import (
	"strconv"
	"strings"
	"unicode"
)

type TranscriptSegment struct {
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Speaker string  `json:"speaker"`
	Text    string  `json:"text"`
}

type transcriptState uint8

const (
	seekStart transcriptState = iota
	readStart
	expectSpeakerOpen
	readSpeaker
	readText
	readEnd
	afterEnd
)

// TranscriptParser incrementally parses [start][Sxx]text[end] without regexes.
type TranscriptParser struct {
	StripText bool
	SkipEmpty bool
	state     transcriptState
	token     []rune
	text      []rune
	pending   []rune
	start     float64
	end       float64
	hasStart  bool
	hasEnd    bool
	endToken  string
	speaker   string
}

func NewTranscriptParser() *TranscriptParser {
	return &TranscriptParser{StripText: true, SkipEmpty: true}
}

func (p *TranscriptParser) Reset() {
	strip, skip := p.StripText, p.SkipEmpty
	*p = TranscriptParser{StripText: strip, SkipEmpty: skip}
}

func (p *TranscriptParser) Feed(chunk string) []TranscriptSegment {
	var out []TranscriptSegment
	for _, ch := range chunk {
		switch p.state {
		case seekStart:
			if ch == '[' {
				p.token = p.token[:0]
				p.state = readStart
			}
		case readStart:
			p.consumeStart(ch)
		case expectSpeakerOpen:
			if ch == '[' {
				p.token = p.token[:0]
				p.state = readSpeaker
			} else if !unicode.IsSpace(ch) {
				p.Reset()
			}
		case readSpeaker:
			p.consumeSpeaker(ch)
		case readText:
			if ch == '[' {
				p.token = p.token[:0]
				p.state = readEnd
			} else {
				p.text = append(p.text, ch)
			}
		case readEnd:
			p.consumeEnd(ch)
		case afterEnd:
			if ch == '[' {
				p.emit(&out)
				p.token = p.token[:0]
				p.state = readStart
			} else if unicode.IsSpace(ch) {
				p.pending = append(p.pending, ch)
			} else {
				p.text = append(p.text, '[')
				p.text = append(p.text, []rune(p.endToken)...)
				p.text = append(p.text, ']')
				p.text = append(p.text, p.pending...)
				p.text = append(p.text, ch)
				p.pending = p.pending[:0]
				p.hasEnd = false
				p.endToken = ""
				p.state = readText
			}
		}
	}
	return out
}

func (p *TranscriptParser) Close() []TranscriptSegment {
	var out []TranscriptSegment
	if p.state == afterEnd {
		p.emit(&out)
	}
	p.Reset()
	return out
}

func ParseTranscript(text string) []TranscriptSegment {
	p := NewTranscriptParser()
	out := p.Feed(text)
	return append(out, p.Close()...)
}

func (p *TranscriptParser) consumeStart(ch rune) {
	if ch == ']' {
		if value, ok := parseTimestamp(p.token); ok {
			p.start, p.hasStart, p.state = value, true, expectSpeakerOpen
			p.token = p.token[:0]
			return
		}
		p.Reset()
		return
	}
	if timestampRune(ch) && len(p.token) < 32 {
		p.token = append(p.token, ch)
		return
	}
	p.Reset()
	if ch == '[' {
		p.state = readStart
	}
}

func (p *TranscriptParser) consumeSpeaker(ch rune) {
	if ch == ']' {
		if speaker, ok := parseSpeaker(p.token); ok {
			p.speaker = speaker
			p.text = p.text[:0]
			p.state = readText
			p.token = p.token[:0]
			return
		}
		p.Reset()
		return
	}
	if (ch == 'S' || ch >= '0' && ch <= '9') && len(p.token) < 16 {
		p.token = append(p.token, ch)
		return
	}
	p.Reset()
	if ch == '[' {
		p.state = readStart
	}
}

func (p *TranscriptParser) consumeEnd(ch rune) {
	if ch == ']' {
		if value, ok := parseTimestamp(p.token); ok && p.hasStart && value >= p.start {
			p.end, p.hasEnd = value, true
			p.endToken = string(p.token)
			p.pending = p.pending[:0]
			p.state = afterEnd
		} else {
			p.text = append(p.text, '[')
			p.text = append(p.text, p.token...)
			p.text = append(p.text, ']')
			p.state = readText
		}
		p.token = p.token[:0]
		return
	}
	if timestampRune(ch) && len(p.token) < 32 {
		p.token = append(p.token, ch)
		return
	}
	p.text = append(p.text, '[')
	p.text = append(p.text, p.token...)
	p.text = append(p.text, ch)
	p.token = p.token[:0]
	p.state = readText
}

func (p *TranscriptParser) emit(out *[]TranscriptSegment) {
	if !p.hasStart || !p.hasEnd || p.speaker == "" {
		p.Reset()
		return
	}
	text := string(p.text)
	if p.StripText {
		text = strings.TrimSpace(text)
	}
	if text != "" || !p.SkipEmpty {
		*out = append(*out, TranscriptSegment{Start: p.start, End: p.end, Speaker: p.speaker, Text: text})
	}
	p.Reset()
}

func parseTimestamp(chars []rune) (float64, bool) {
	if len(chars) == 0 {
		return 0, false
	}
	dots, digits := 0, 0
	for _, ch := range chars {
		switch {
		case ch >= '0' && ch <= '9':
			digits++
		case ch == '.':
			dots++
			if dots > 1 {
				return 0, false
			}
		default:
			return 0, false
		}
	}
	if digits == 0 {
		return 0, false
	}
	value, err := strconv.ParseFloat(string(chars), 64)
	return value, err == nil
}

func parseSpeaker(chars []rune) (string, bool) {
	if len(chars) < 2 || chars[0] != 'S' {
		return "", false
	}
	for _, ch := range chars[1:] {
		if ch < '0' || ch > '9' {
			return "", false
		}
	}
	return string(chars), true
}

func timestampRune(ch rune) bool { return ch == '.' || ch >= '0' && ch <= '9' }
