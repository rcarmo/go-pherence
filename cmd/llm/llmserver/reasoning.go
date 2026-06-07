package main

import "strings"

// reasoningSplitter separates Qwen-style <think>...</think> text from answer text.
// It is deliberately streaming-friendly: tokens before </think> are treated as
// reasoning once a <think> opener has been observed, and answer text resumes
// after the closer. Plain text with no opener remains content.
type reasoningSplitter struct {
	inReasoning bool
	seenThink   bool
	pending     string
}

func splitReasoningText(text string) (content, reasoning string) {
	var s reasoningSplitter
	c, r := s.Push(text)
	pc, pr := s.Flush()
	return c + pc, r + pr
}

func (s *reasoningSplitter) Push(text string) (content, reasoning string) {
	s.pending += text
	var cb, rb strings.Builder
	for len(s.pending) > 0 {
		if s.inReasoning {
			end := strings.Index(s.pending, "</think>")
			if end < 0 {
				rb.WriteString(s.pending)
				s.pending = ""
				break
			}
			rb.WriteString(s.pending[:end])
			s.pending = s.pending[end+len("</think>"):]
			s.inReasoning = false
			continue
		}

		start := strings.Index(s.pending, "<think>")
		if start < 0 {
			// Keep a possible partial opener buffered across token boundaries.
			keep := longestThinkPrefixSuffix(s.pending)
			if keep == len(s.pending) {
				break
			}
			cb.WriteString(s.pending[:len(s.pending)-keep])
			s.pending = s.pending[len(s.pending)-keep:]
			break
		}
		cb.WriteString(s.pending[:start])
		s.pending = s.pending[start+len("<think>"):]
		s.inReasoning = true
		s.seenThink = true
	}
	return cb.String(), rb.String()
}

func (s *reasoningSplitter) Flush() (content, reasoning string) {
	if s.pending == "" {
		return "", ""
	}
	p := s.pending
	s.pending = ""
	if s.inReasoning || s.seenThink {
		return "", p
	}
	return p, ""
}

func longestThinkPrefixSuffix(s string) int {
	const marker = "<think>"
	max := len(marker) - 1
	if len(s) < max {
		max = len(s)
	}
	for n := max; n > 0; n-- {
		if strings.HasSuffix(s, marker[:n]) {
			return n
		}
	}
	return 0
}
