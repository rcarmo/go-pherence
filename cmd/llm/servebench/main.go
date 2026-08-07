package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/rcarmo/go-pherence/runtime/servingbench"
)

func main() {
	endpoint := flag.String("endpoint", "http://127.0.0.1:8080/v1/chat/completions", "OpenAI-compatible streaming chat endpoint")
	model := flag.String("model", "", "model ID")
	concurrencyCSV := flag.String("concurrency", "1,2,4,8", "comma-separated concurrency levels")
	requests := flag.Int("requests", 16, "requests per concurrency level")
	promptsCSV := flag.String("prompts", "Hello|Explain paged attention briefly.|Write a Go function that adds two integers.", "pipe-separated prompts")
	promptFile := flag.String("prompt-file", "", "optional newline-delimited prompt file")
	maxTokens := flag.Int("max-tokens", 32, "maximum output tokens")
	arrival := flag.String("arrival", "fixed", "arrival mode: fixed, poisson, gamma")
	rate := flag.Float64("rate", 2, "mean request arrival rate per second")
	gammaShape := flag.Float64("gamma-shape", 2, "Gamma arrival shape")
	seed := flag.Int64("seed", 1, "arrival RNG seed")
	timeout := flag.Duration("timeout", 5*time.Minute, "per-request timeout")
	ttftSLO := flag.Duration("ttft-slo", 0, "optional TTFT SLO")
	itlSLO := flag.Duration("itl-slo", 0, "optional ITL SLO")
	e2eSLO := flag.Duration("e2e-slo", 0, "optional E2E SLO")
	jsonPath := flag.String("json", "", "write full JSON report to path (stdout when empty)")
	csvPath := flag.String("csv", "", "write summary CSV to path")
	flag.Parse()

	concurrencies, err := parsePositiveInts(*concurrencyCSV)
	fatalIf(err)
	prompts, err := loadPrompts(*promptsCSV, *promptFile)
	fatalIf(err)

	reports := make([]servingbench.Report, 0, len(concurrencies))
	for _, concurrency := range concurrencies {
		report, err := servingbench.Run(context.Background(), servingbench.Config{
			Endpoint: *endpoint, Model: *model, Prompts: prompts,
			RequestCount: *requests, Concurrency: concurrency, MaxTokens: *maxTokens,
			Seed: *seed, Timeout: *timeout,
			Arrival: servingbench.ArrivalConfig{Mode: servingbench.ArrivalMode(*arrival), Rate: *rate, GammaShape: *gammaShape},
			SLO:     servingbench.SLOConfig{TTFT: *ttftSLO, ITL: *itlSLO, E2E: *e2eSLO},
		})
		fatalIf(err)
		reports = append(reports, report)
	}

	var out = os.Stdout
	if *jsonPath != "" {
		f, err := os.Create(*jsonPath)
		fatalIf(err)
		defer f.Close()
		out = f
	}
	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	fatalIf(enc.Encode(reports))
	if *csvPath != "" {
		f, err := os.Create(*csvPath)
		fatalIf(err)
		defer f.Close()
		fatalIf(servingbench.WriteSummaryCSV(f, reports))
	}
}

func parsePositiveInts(s string) ([]int, error) {
	var out []int
	seen := map[int]bool{}
	for _, field := range strings.Split(s, ",") {
		field = strings.TrimSpace(field)
		if field == "" {
			continue
		}
		v, err := strconv.Atoi(field)
		if err != nil || v <= 0 {
			return nil, fmt.Errorf("invalid positive integer %q", field)
		}
		if !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no concurrency levels")
	}
	return out, nil
}

func loadPrompts(inline, path string) ([]string, error) {
	if path != "" {
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		inline = strings.ReplaceAll(string(b), "\r\n", "\n")
		inline = strings.ReplaceAll(inline, "\n", "|")
	}
	var out []string
	for _, p := range strings.Split(inline, "|") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no prompts")
	}
	return out, nil
}

func fatalIf(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "servebench:", err)
		os.Exit(1)
	}
}
