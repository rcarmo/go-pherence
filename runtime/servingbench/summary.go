package servingbench

import (
	"encoding/csv"
	"io"
	"sort"
	"strconv"
	"time"
)

// Percentiles holds p50/p95/p99 values for a duration sample.
type Percentiles struct {
	Count int           `json:"count"`
	P50   time.Duration `json:"p50"`
	P95   time.Duration `json:"p95"`
	P99   time.Duration `json:"p99"`
}

// Summary aggregates one benchmark run.
type Summary struct {
	Requests              int           `json:"requests"`
	Successful            int           `json:"successful"`
	Errors                int           `json:"errors"`
	Cancelled             int           `json:"cancelled"`
	UsageRequests         int           `json:"usage_requests"`
	StatusCodes           map[int]int   `json:"status_codes,omitempty"`
	Duration              time.Duration `json:"duration"`
	QueueTime             Percentiles   `json:"queue_time"`
	TTFT                  Percentiles   `json:"ttft"`
	ITL                   Percentiles   `json:"itl"`
	TPOT                  Percentiles   `json:"tpot"`
	E2E                   Percentiles   `json:"e2e"`
	InputTokens           int           `json:"input_tokens"`
	OutputTokens          int           `json:"output_tokens"`
	TotalTokens           int           `json:"total_tokens"`
	InputTokensPerSecond  float64       `json:"input_tokens_per_second"`
	OutputTokensPerSecond float64       `json:"output_tokens_per_second"`
	TotalTokensPerSecond  float64       `json:"total_tokens_per_second"`
	RequestsPerSecond     float64       `json:"requests_per_second"`
	GoodRequests          int           `json:"good_requests"`
	GoodputPerSecond      float64       `json:"goodput_per_second"`
	GoodputRatio          float64       `json:"goodput_ratio"`
}

// Percentile returns the linear-interpolated percentile for q in [0,1].
func Percentile(values []time.Duration, q float64) time.Duration {
	if len(values) == 0 {
		return 0
	}
	if q <= 0 {
		return minDuration(values)
	}
	if q >= 1 {
		return maxDuration(values)
	}
	ordered := append([]time.Duration(nil), values...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i] < ordered[j] })
	if len(ordered) == 1 {
		return ordered[0]
	}
	pos := q * float64(len(ordered)-1)
	lo := int(pos)
	hi := lo
	if lo < len(ordered)-1 {
		hi = lo + 1
	}
	frac := pos - float64(lo)
	base := float64(ordered[lo])
	span := float64(ordered[hi] - ordered[lo])
	return time.Duration(base + frac*span)
}

func buildPercentiles(values []time.Duration) Percentiles {
	return Percentiles{
		Count: len(values),
		P50:   Percentile(values, 0.50),
		P95:   Percentile(values, 0.95),
		P99:   Percentile(values, 0.99),
	}
}

func minDuration(values []time.Duration) time.Duration {
	min := values[0]
	for _, v := range values[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

func maxDuration(values []time.Duration) time.Duration {
	max := values[0]
	for _, v := range values[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

// Summarize aggregates per-request results into headline metrics.
func Summarize(requests []RequestResult, totalDuration time.Duration, slo SLOConfig) Summary {
	summary := Summary{
		Requests:    len(requests),
		StatusCodes: make(map[int]int),
		Duration:    totalDuration,
	}
	var queueTimes, ttfts, itls, tpots, e2es []time.Duration
	for _, req := range requests {
		if req.StatusCode != 0 {
			summary.StatusCodes[req.StatusCode]++
		}
		if req.Cancelled {
			summary.Cancelled++
		} else if req.Error != "" || req.StatusCode >= 400 {
			summary.Errors++
		}
		if req.Successful() {
			summary.Successful++
		}
		if req.PromptTokens != nil || req.CompletionTokens != nil || req.TotalTokens != nil {
			summary.UsageRequests++
		}
		summary.InputTokens += req.InputTokenCount()
		summary.OutputTokens += req.OutputTokenCount()
		summary.TotalTokens += req.TotalTokenCount()
		if v, ok := req.QueueTime(); ok {
			queueTimes = append(queueTimes, v)
		}
		if v, ok := req.TTFT(); ok {
			ttfts = append(ttfts, v)
		}
		if v, ok := req.TPOT(); ok {
			tpots = append(tpots, v)
		}
		if v, ok := req.E2E(); ok {
			e2es = append(e2es, v)
		}
		itls = append(itls, req.ITLs()...)
		if req.MeetsSLO(slo) {
			summary.GoodRequests++
		}
	}
	if len(summary.StatusCodes) == 0 {
		summary.StatusCodes = nil
	}
	seconds := totalDuration.Seconds()
	if seconds > 0 {
		summary.InputTokensPerSecond = float64(summary.InputTokens) / seconds
		summary.OutputTokensPerSecond = float64(summary.OutputTokens) / seconds
		summary.TotalTokensPerSecond = float64(summary.TotalTokens) / seconds
		summary.RequestsPerSecond = float64(summary.Requests) / seconds
		summary.GoodputPerSecond = float64(summary.GoodRequests) / seconds
	}
	if summary.Requests > 0 {
		summary.GoodputRatio = float64(summary.GoodRequests) / float64(summary.Requests)
	}
	summary.QueueTime = buildPercentiles(queueTimes)
	summary.TTFT = buildPercentiles(ttfts)
	summary.ITL = buildPercentiles(itls)
	summary.TPOT = buildPercentiles(tpots)
	summary.E2E = buildPercentiles(e2es)
	return summary
}

// WriteSummaryCSV writes one summary row per report.
func WriteSummaryCSV(w io.Writer, reports []Report) error {
	cw := csv.NewWriter(w)
	rows := [][]string{{
		"endpoint",
		"model",
		"arrival_mode",
		"arrival_rate",
		"gamma_shape",
		"concurrency",
		"requests",
		"prompt_count",
		"max_tokens",
		"seed",
		"timeout_ms",
		"duration_ms",
		"successful",
		"errors",
		"cancelled",
		"usage_requests",
		"good_requests",
		"goodput_rps",
		"goodput_ratio",
		"requests_per_sec",
		"input_tokens",
		"output_tokens",
		"total_tokens",
		"input_tokens_per_sec",
		"output_tokens_per_sec",
		"total_tokens_per_sec",
		"queue_p50_ms",
		"queue_p95_ms",
		"queue_p99_ms",
		"ttft_p50_ms",
		"ttft_p95_ms",
		"ttft_p99_ms",
		"itl_p50_ms",
		"itl_p95_ms",
		"itl_p99_ms",
		"tpot_p50_ms",
		"tpot_p95_ms",
		"tpot_p99_ms",
		"e2e_p50_ms",
		"e2e_p95_ms",
		"e2e_p99_ms",
	}}
	for _, report := range reports {
		s := report.Summary
		cfg := report.Config
		rows = append(rows, []string{
			cfg.Endpoint,
			cfg.Model,
			string(cfg.Arrival.normalized().Mode),
			formatFloat(cfg.Arrival.Rate),
			formatFloat(cfg.Arrival.normalized().GammaShape),
			strconv.Itoa(cfg.Concurrency),
			strconv.Itoa(cfg.RequestCount),
			strconv.Itoa(len(cfg.Prompts)),
			strconv.Itoa(cfg.MaxTokens),
			strconv.FormatInt(cfg.Seed, 10),
			formatMillis(cfg.Timeout),
			formatMillis(s.Duration),
			strconv.Itoa(s.Successful),
			strconv.Itoa(s.Errors),
			strconv.Itoa(s.Cancelled),
			strconv.Itoa(s.UsageRequests),
			strconv.Itoa(s.GoodRequests),
			formatFloat(s.GoodputPerSecond),
			formatFloat(s.GoodputRatio),
			formatFloat(s.RequestsPerSecond),
			strconv.Itoa(s.InputTokens),
			strconv.Itoa(s.OutputTokens),
			strconv.Itoa(s.TotalTokens),
			formatFloat(s.InputTokensPerSecond),
			formatFloat(s.OutputTokensPerSecond),
			formatFloat(s.TotalTokensPerSecond),
			formatMillis(s.QueueTime.P50),
			formatMillis(s.QueueTime.P95),
			formatMillis(s.QueueTime.P99),
			formatMillis(s.TTFT.P50),
			formatMillis(s.TTFT.P95),
			formatMillis(s.TTFT.P99),
			formatMillis(s.ITL.P50),
			formatMillis(s.ITL.P95),
			formatMillis(s.ITL.P99),
			formatMillis(s.TPOT.P50),
			formatMillis(s.TPOT.P95),
			formatMillis(s.TPOT.P99),
			formatMillis(s.E2E.P50),
			formatMillis(s.E2E.P95),
			formatMillis(s.E2E.P99),
		})
	}
	if err := cw.WriteAll(rows); err != nil {
		return err
	}
	cw.Flush()
	return cw.Error()
}

func formatMillis(d time.Duration) string {
	return strconv.FormatFloat(float64(d)/float64(time.Millisecond), 'f', 3, 64)
}

func formatFloat(v float64) string {
	return strconv.FormatFloat(v, 'f', 6, 64)
}
