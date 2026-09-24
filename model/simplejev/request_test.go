package simplejev

import (
	"strings"
	"testing"
)

const validRequest = `{"state":"Evidence A","questions":[{"id":"q1","instruction":"Choose one","kind":"choice","labels":["a","b"]}]}`

func TestDecodeRequest(t *testing.T) {
	req, err := DecodeRequest(strings.NewReader(validRequest))
	if err != nil || req.State != "Evidence A" || len(req.Questions) != 1 || req.Questions[0].Labels[0] != "a" {
		t.Fatalf("decoded request=%+v err=%v", req, err)
	}
}

func TestDecodeRequestRejectsMalformed(t *testing.T) {
	cases := map[string]string{
		"empty":               "",
		"unknown top":         `{"state":"x","questions":[],"extra":1}`,
		"case variant top":    `{"State":"x","questions":[{"id":"q","instruction":"x","kind":"choice","labels":["a","b"]}]}`,
		"duplicate top":       `{"state":"x","state":"y","questions":[{"id":"q","instruction":"x","kind":"choice","labels":["a","b"]}]}`,
		"duplicate nested":    `{"state":"x","questions":[{"id":"q","id":"z","instruction":"x","kind":"choice","labels":["a","b"]}]}`,
		"case variant nested": `{"state":"x","questions":[{"ID":"q","instruction":"x","kind":"choice","labels":["a","b"]}]}`,
		"unknown question":    `{"state":"x","questions":[{"id":"q","instruction":"x","kind":"choice","labels":["a","b"],"extra":1}]}`,
		"trailing object":     validRequest + validRequest,
		"trailing garbage":    validRequest + ` !`,
		"empty state":         strings.Replace(validRequest, "Evidence A", "  ", 1),
		"invisible state":     strings.Replace(validRequest, "Evidence A", "\u200b", 1),
		"invisible ID":        strings.Replace(validRequest, `"q1"`, `"\u200b"`, 1),
		"space ID":            strings.Replace(validRequest, `"q1"`, `" "`, 1),
		"invisible label":     strings.Replace(validRequest, `"a"`, `"\u200b"`, 1),
		"duplicate question":  `{"state":"x","questions":[{"id":"q","instruction":"x","kind":"choice","labels":["a","b"]},{"id":"q","instruction":"x","kind":"choice","labels":["a","b"]}]}`,
		"duplicate label":     strings.Replace(validRequest, `["a","b"]`, `["a","a"]`, 1),
		"bad kind":            strings.Replace(validRequest, `"choice"`, `"custom"`, 1),
		"too many labels":     strings.Replace(validRequest, `["a","b"]`, `["a","b","`+strings.Repeat(`z","`, 49)+`x"]`, 1),
		"short labels":        strings.Replace(validRequest, `["a","b"]`, `["a"]`, 1),
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := DecodeRequest(strings.NewReader(data)); err == nil {
				t.Fatal("accepted malformed request")
			}
		})
	}
	if _, err := DecodeRequest(nil); err == nil {
		t.Fatal("accepted nil reader")
	}
	if _, err := DecodeRequest(strings.NewReader(strings.Repeat("x", MaxRequestBytes+1))); err == nil {
		t.Fatal("accepted oversized body")
	}
}

func TestRequestValidateBounds(t *testing.T) {
	base := Request{State: "s", Questions: []Question{{ID: "q", Instruction: "choose", Kind: "ordinal", Labels: []string{"a", "b"}}}}
	if err := base.Validate(); err != nil {
		t.Fatal(err)
	}
	base.State = strings.Repeat("x", MaxStateBytes+1)
	if err := base.Validate(); err == nil {
		t.Fatal("accepted oversized state")
	}
	base.State = "s"
	base.Questions[0].Instruction = strings.Repeat("x", MaxInstructionBytes+1)
	if err := base.Validate(); err == nil {
		t.Fatal("accepted oversized instruction")
	}
	base.Questions[0].Instruction = "choose"
	base.Questions[0].ID = strings.Repeat("q", 129)
	if err := base.Validate(); err == nil {
		t.Fatal("accepted oversized question ID")
	}
	base.Questions[0].ID = "q"
	base.Questions[0].Labels = []string{strings.Repeat("a", 257), "b"}
	if err := base.Validate(); err == nil {
		t.Fatal("accepted oversized label")
	}
	base.Questions[0].Labels = []string{"a", "b"}
	base.Questions[0].Labels[0] = strings.Repeat("é", 129) // 258 UTF-8 bytes.
	if err := base.Validate(); err == nil {
		t.Fatal("accepted multibyte label beyond byte ceiling")
	}
	base.Questions[0].Labels[0] = "a"
	base.Questions = append(base.Questions, make([]Question, MaxQuestions)...)
	if err := base.Validate(); err == nil {
		t.Fatal("accepted too many questions")
	}
}
