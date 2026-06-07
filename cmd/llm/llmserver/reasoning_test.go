package main

import "testing"

func TestSplitReasoningText(t *testing.T) {
	content, reasoning := splitReasoningText("<think>plan first</think>answer")
	if content != "answer" || reasoning != "plan first" {
		t.Fatalf("content=%q reasoning=%q", content, reasoning)
	}
}

func TestReasoningSplitterAcrossChunks(t *testing.T) {
	var s reasoningSplitter
	c, r := s.Push("<thi")
	if c != "" || r != "" {
		t.Fatalf("partial opener emitted content=%q reasoning=%q", c, r)
	}
	c, r = s.Push("nk>step")
	if c != "" || r != "step" {
		t.Fatalf("reasoning chunk content=%q reasoning=%q", c, r)
	}
	c, r = s.Push("</think>done")
	if c != "done" || r != "" {
		t.Fatalf("answer chunk content=%q reasoning=%q", c, r)
	}
}

func TestSplitReasoningTextPlainContent(t *testing.T) {
	content, reasoning := splitReasoningText("plain answer")
	if content != "plain answer" || reasoning != "" {
		t.Fatalf("content=%q reasoning=%q", content, reasoning)
	}
}
