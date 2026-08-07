package servingbench

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
)

// SLOConfig defines optional latency SLOs used to compute goodput.
type SLOConfig struct {
	TTFT time.Duration `json:"ttft,omitempty"`
	ITL  time.Duration `json:"itl,omitempty"`
	E2E  time.Duration `json:"e2e,omitempty"`
}

// Config describes a single serving benchmark run at a fixed concurrency.
type Config struct {
	Endpoint     string        `json:"endpoint"`
	Model        string        `json:"model,omitempty"`
	Prompts      []string      `json:"prompts"`
	RequestCount int           `json:"request_count"`
	Concurrency  int           `json:"concurrency"`
	MaxTokens    int           `json:"max_tokens"`
	Seed         int64         `json:"seed"`
	Timeout      time.Duration `json:"timeout"`
	Arrival      ArrivalConfig `json:"arrival"`
	SLO          SLOConfig     `json:"slo"`
	HTTPClient   *http.Client  `json:"-"`
}

// RequestResult records one benchmarked request.
type RequestResult struct {
	Index                  int             `json:"index"`
	PromptIndex            int             `json:"prompt_index"`
	Prompt                 string          `json:"prompt"`
	ScheduledArrivalOffset time.Duration   `json:"scheduled_arrival_offset"`
	ArrivalOffset          *time.Duration  `json:"arrival_offset,omitempty"`
	StartOffset            *time.Duration  `json:"start_offset,omitempty"`
	FirstTokenOffset       *time.Duration  `json:"first_token_offset,omitempty"`
	LastTokenOffset        *time.Duration  `json:"last_token_offset,omitempty"`
	TokenOffsets           []time.Duration `json:"token_offsets,omitempty"`
	EndOffset              *time.Duration  `json:"end_offset,omitempty"`
	Status                 string          `json:"status"`
	StatusCode             int             `json:"status_code,omitempty"`
	Cancelled              bool            `json:"cancelled,omitempty"`
	Error                  string          `json:"error,omitempty"`
	FinishReason           string          `json:"finish_reason,omitempty"`
	PromptTokens           *int            `json:"prompt_tokens,omitempty"`
	CompletionTokens       *int            `json:"completion_tokens,omitempty"`
	TotalTokens            *int            `json:"total_tokens,omitempty"`
}

// Report is one benchmark result for a single concurrency level.
type Report struct {
	Config     Config          `json:"config"`
	StartedAt  time.Time       `json:"started_at"`
	FinishedAt time.Time       `json:"finished_at"`
	Requests   []RequestResult `json:"requests"`
	Summary    Summary         `json:"summary"`
}

func (cfg Config) normalized() Config {
	cfg.Arrival = cfg.Arrival.normalized()
	return cfg
}

// Validate checks the benchmark configuration.
func (cfg Config) Validate() error {
	cfg = cfg.normalized()
	if cfg.Endpoint == "" {
		return fmt.Errorf("endpoint is required")
	}
	if len(cfg.Prompts) == 0 {
		return fmt.Errorf("at least one prompt is required")
	}
	if cfg.RequestCount <= 0 {
		return fmt.Errorf("request count must be positive")
	}
	if cfg.Concurrency <= 0 {
		return fmt.Errorf("concurrency must be positive")
	}
	if cfg.MaxTokens < 0 {
		return fmt.Errorf("max tokens must be non-negative")
	}
	if cfg.Timeout < 0 {
		return fmt.Errorf("timeout must be non-negative")
	}
	return cfg.Arrival.Validate()
}

// Duration returns the elapsed wall time for the run.
func (r Report) Duration() time.Duration {
	if r.FinishedAt.IsZero() || r.StartedAt.IsZero() {
		return 0
	}
	return r.FinishedAt.Sub(r.StartedAt)
}

// QueueTime returns the client-side queueing delay from arrival to start.
func (r RequestResult) QueueTime() (time.Duration, bool) {
	if r.ArrivalOffset == nil || r.StartOffset == nil {
		return 0, false
	}
	return *r.StartOffset - *r.ArrivalOffset, true
}

// TTFT returns time-to-first-token from request start.
func (r RequestResult) TTFT() (time.Duration, bool) {
	if r.StartOffset == nil || r.FirstTokenOffset == nil {
		return 0, false
	}
	return *r.FirstTokenOffset - *r.StartOffset, true
}

// E2E returns end-to-end latency from arrival to request completion.
func (r RequestResult) E2E() (time.Duration, bool) {
	if r.ArrivalOffset == nil || r.EndOffset == nil {
		return 0, false
	}
	return *r.EndOffset - *r.ArrivalOffset, true
}

// ITLs returns the per-token inter-arrival latencies for the request stream.
func (r RequestResult) ITLs() []time.Duration {
	if len(r.TokenOffsets) < 2 {
		return nil
	}
	out := make([]time.Duration, 0, len(r.TokenOffsets)-1)
	for i := 1; i < len(r.TokenOffsets); i++ {
		out = append(out, r.TokenOffsets[i]-r.TokenOffsets[i-1])
	}
	return out
}

// TPOT returns average time per emitted token after the first stream token.
func (r RequestResult) TPOT() (time.Duration, bool) {
	if len(r.TokenOffsets) < 2 {
		return 0, false
	}
	elapsed := r.TokenOffsets[len(r.TokenOffsets)-1] - r.TokenOffsets[0]
	return elapsed / time.Duration(len(r.TokenOffsets)-1), true
}

// Successful reports whether the request completed with a successful HTTP status
// and no client-side error or cancellation.
func (r RequestResult) Successful() bool {
	return !r.Cancelled && r.Error == "" && r.StatusCode >= 200 && r.StatusCode < 300
}

// MeetsSLO reports whether the request is successful and satisfies the supplied
// TTFT/ITL/E2E SLOs.
func (r RequestResult) MeetsSLO(slo SLOConfig) bool {
	if !r.Successful() {
		return false
	}
	if slo.TTFT > 0 {
		v, ok := r.TTFT()
		if !ok || v > slo.TTFT {
			return false
		}
	}
	if slo.ITL > 0 {
		for _, itl := range r.ITLs() {
			if itl > slo.ITL {
				return false
			}
		}
	}
	if slo.E2E > 0 {
		v, ok := r.E2E()
		if !ok || v > slo.E2E {
			return false
		}
	}
	return true
}

// InputTokenCount returns prompt tokens when the server exposed them.
func (r RequestResult) InputTokenCount() int {
	if r.PromptTokens == nil {
		return 0
	}
	return *r.PromptTokens
}

// OutputTokenCount returns completion tokens when exposed by the server and
// otherwise falls back to the number of streamed token events observed.
func (r RequestResult) OutputTokenCount() int {
	if r.CompletionTokens != nil {
		return *r.CompletionTokens
	}
	return len(r.TokenOffsets)
}

// TotalTokenCount returns total tokens when exposed by the server and otherwise
// uses the available input/output counts.
func (r RequestResult) TotalTokenCount() int {
	if r.TotalTokens != nil {
		return *r.TotalTokens
	}
	return r.InputTokenCount() + r.OutputTokenCount()
}

// Run executes a single benchmark run.
func Run(ctx context.Context, cfg Config) (Report, error) {
	cfg = cfg.normalized()
	if err := cfg.Validate(); err != nil {
		return Report{}, err
	}
	offsets, err := GenerateArrivalOffsets(cfg.RequestCount, cfg.Arrival, cfg.Seed)
	if err != nil {
		return Report{}, err
	}
	report := Report{
		Config:    cfg,
		StartedAt: time.Now(),
		Requests:  make([]RequestResult, cfg.RequestCount),
	}
	for i := range report.Requests {
		report.Requests[i] = RequestResult{
			Index:                  i,
			PromptIndex:            i % len(cfg.Prompts),
			Prompt:                 cfg.Prompts[i%len(cfg.Prompts)],
			ScheduledArrivalOffset: offsets[i],
			Status:                 "pending",
		}
	}

	jobs := make(chan int, cfg.RequestCount)
	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	var wg sync.WaitGroup
	for worker := 0; worker < cfg.Concurrency; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				if err := ctx.Err(); err != nil {
					cancelBeforeStart(&report.Requests[idx], err)
					continue
				}
				runRequest(ctx, client, cfg, report.StartedAt, &report.Requests[idx])
			}
		}()
	}

scheduleLoop:
	for i := range report.Requests {
		if err := waitUntil(ctx, report.StartedAt.Add(report.Requests[i].ScheduledArrivalOffset)); err != nil {
			markPendingCancelled(report.Requests[i:], err)
			break scheduleLoop
		}
		arrival := time.Since(report.StartedAt)
		report.Requests[i].ArrivalOffset = durationPtr(arrival)
		report.Requests[i].Status = "queued"
		select {
		case jobs <- i:
		case <-ctx.Done():
			cancelBeforeStart(&report.Requests[i], ctx.Err())
			markPendingCancelled(report.Requests[i+1:], ctx.Err())
			break scheduleLoop
		}
	}
	close(jobs)
	wg.Wait()
	report.FinishedAt = time.Now()
	report.Summary = Summarize(report.Requests, report.Duration(), cfg.SLO)
	return report, nil
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatCompletionRequest struct {
	Model       string        `json:"model,omitempty"`
	Messages    []chatMessage `json:"messages"`
	Stream      bool          `json:"stream"`
	Temperature int           `json:"temperature"`
	MaxTokens   int           `json:"max_tokens"`
}

func runRequest(parent context.Context, client *http.Client, cfg Config, startedAt time.Time, result *RequestResult) {
	start := time.Since(startedAt)
	result.StartOffset = durationPtr(start)
	result.Status = "started"

	ctx := parent
	cancel := func() {}
	if cfg.Timeout > 0 {
		ctx, cancel = context.WithTimeout(parent, cfg.Timeout)
	} else {
		ctx, cancel = context.WithCancel(parent)
	}
	defer cancel()
	defer finalizeRequest(result, startedAt)

	payload, err := json.Marshal(chatCompletionRequest{
		Model: cfg.Model,
		Messages: []chatMessage{{
			Role:    "user",
			Content: result.Prompt,
		}},
		Stream:      true,
		Temperature: 0,
		MaxTokens:   cfg.MaxTokens,
	})
	if err != nil {
		setRequestError(result, ctx, err)
		return
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.Endpoint, bytes.NewReader(payload))
	if err != nil {
		setRequestError(result, ctx, err)
		return
	}
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		setRequestError(result, ctx, err)
		return
	}
	defer resp.Body.Close()
	result.StatusCode = resp.StatusCode
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
		bodyText := strings.TrimSpace(string(body))
		if bodyText == "" {
			bodyText = http.StatusText(resp.StatusCode)
		}
		setRequestError(result, ctx, fmt.Errorf("unexpected HTTP status %d: %s", resp.StatusCode, bodyText))
		return
	}

	err = ParseChatCompletionStream(resp.Body, func(chunk ChatCompletionChunk) error {
		if chunk.Usage != nil {
			applyUsage(result, *chunk.Usage)
		}
		if chunk.FinishReason != "" {
			result.FinishReason = chunk.FinishReason
		}
		if chunk.Done {
			return nil
		}
		if chunk.Content == "" && chunk.ReasoningContent == "" {
			return nil
		}
		offset := time.Since(startedAt)
		result.TokenOffsets = append(result.TokenOffsets, offset)
		if result.FirstTokenOffset == nil {
			result.FirstTokenOffset = durationPtr(offset)
		}
		result.LastTokenOffset = durationPtr(offset)
		return nil
	})
	if err != nil {
		setRequestError(result, ctx, err)
	}
}

func finalizeRequest(result *RequestResult, startedAt time.Time) {
	end := time.Since(startedAt)
	result.EndOffset = durationPtr(end)
	if result.CompletionTokens == nil && len(result.TokenOffsets) > 0 {
		value := len(result.TokenOffsets)
		result.CompletionTokens = intPtr(value)
	}
	if result.TotalTokens == nil && result.PromptTokens != nil && result.CompletionTokens != nil {
		value := *result.PromptTokens + *result.CompletionTokens
		result.TotalTokens = intPtr(value)
	}
	switch {
	case result.Cancelled:
		result.Status = "canceled"
	case result.Error != "":
		if result.StatusCode >= 400 {
			result.Status = "http_error"
		} else {
			result.Status = "error"
		}
	case result.StatusCode >= 200 && result.StatusCode < 300:
		result.Status = "ok"
	case result.StatusCode != 0:
		result.Status = "http_error"
	default:
		result.Status = "error"
	}
}

func applyUsage(result *RequestResult, usage Usage) {
	result.PromptTokens = intPtr(usage.PromptTokens)
	result.CompletionTokens = intPtr(usage.CompletionTokens)
	result.TotalTokens = intPtr(usage.TotalTokens)
}

func setRequestError(result *RequestResult, ctx context.Context, err error) {
	if err == nil {
		return
	}
	if ctx != nil && ctx.Err() != nil {
		err = ctx.Err()
	}
	if isCancellationError(err) {
		result.Cancelled = true
	}
	result.Error = err.Error()
}

func isCancellationError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}
	msg := err.Error()
	return strings.Contains(msg, "context canceled") || strings.Contains(msg, "context deadline exceeded")
}

func waitUntil(ctx context.Context, t time.Time) error {
	d := time.Until(t)
	if d <= 0 {
		return nil
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func markPendingCancelled(results []RequestResult, err error) {
	for i := range results {
		if results[i].Status != "pending" {
			continue
		}
		cancelBeforeStart(&results[i], err)
	}
}

func cancelBeforeStart(result *RequestResult, err error) {
	if result == nil {
		return
	}
	result.Cancelled = true
	result.Error = errorString(err)
	result.Status = "canceled"
}

func errorString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func durationPtr(v time.Duration) *time.Duration {
	return &v
}

func intPtr(v int) *int {
	return &v
}
