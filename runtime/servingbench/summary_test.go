package servingbench

import (
	"bytes"
	"encoding/csv"
	"reflect"
	"strconv"
	"testing"
	"time"
)

func TestPercentileInterpolation(t *testing.T) {
	values := []time.Duration{10 * time.Millisecond, 20 * time.Millisecond, 40 * time.Millisecond}
	if got := Percentile(values, -1); got != 10*time.Millisecond {
		t.Fatalf("q<0 percentile = %v, want 10ms", got)
	}
	if got := Percentile(values, 0.50); got != 20*time.Millisecond {
		t.Fatalf("p50 = %v, want 20ms", got)
	}
	if got := Percentile(values, 0.25); got != 15*time.Millisecond {
		t.Fatalf("p25 = %v, want 15ms", got)
	}
	if got := Percentile(values, 0.95); got != 38*time.Millisecond {
		t.Fatalf("p95 = %v, want 38ms", got)
	}
	if got := Percentile(values, 2); got != 40*time.Millisecond {
		t.Fatalf("q>1 percentile = %v, want 40ms", got)
	}
}

func TestRequestResultMeetsSLO(t *testing.T) {
	req := RequestResult{
		StatusCode:       200,
		StartOffset:      durationTestPtr(10 * time.Millisecond),
		FirstTokenOffset: durationTestPtr(30 * time.Millisecond),
		TokenOffsets:     []time.Duration{30 * time.Millisecond, 40 * time.Millisecond, 50 * time.Millisecond},
		ArrivalOffset:    durationTestPtr(0),
		EndOffset:        durationTestPtr(60 * time.Millisecond),
	}
	if !req.MeetsSLO(SLOConfig{TTFT: 25 * time.Millisecond, ITL: 15 * time.Millisecond, E2E: 70 * time.Millisecond}) {
		t.Fatal("request should meet SLOs")
	}
	if req.MeetsSLO(SLOConfig{TTFT: 15 * time.Millisecond}) {
		t.Fatal("request unexpectedly met TTFT SLO")
	}
	if req.MeetsSLO(SLOConfig{ITL: 5 * time.Millisecond}) {
		t.Fatal("request unexpectedly met ITL SLO")
	}
	if req.MeetsSLO(SLOConfig{E2E: 50 * time.Millisecond}) {
		t.Fatal("request unexpectedly met E2E SLO")
	}
}

func TestSummarizeAndWriteSummaryCSV(t *testing.T) {
	requests := []RequestResult{
		{
			Index:            0,
			StatusCode:       200,
			Status:           "ok",
			ArrivalOffset:    durationTestPtr(0),
			StartOffset:      durationTestPtr(10 * time.Millisecond),
			FirstTokenOffset: durationTestPtr(30 * time.Millisecond),
			LastTokenOffset:  durationTestPtr(50 * time.Millisecond),
			TokenOffsets:     []time.Duration{30 * time.Millisecond, 40 * time.Millisecond, 50 * time.Millisecond},
			EndOffset:        durationTestPtr(60 * time.Millisecond),
			PromptTokens:     intTestPtr(10),
			CompletionTokens: intTestPtr(3),
			TotalTokens:      intTestPtr(13),
		},
		{
			Index:            1,
			StatusCode:       200,
			Status:           "ok",
			ArrivalOffset:    durationTestPtr(5 * time.Millisecond),
			StartOffset:      durationTestPtr(25 * time.Millisecond),
			FirstTokenOffset: durationTestPtr(65 * time.Millisecond),
			LastTokenOffset:  durationTestPtr(80 * time.Millisecond),
			TokenOffsets:     []time.Duration{65 * time.Millisecond, 80 * time.Millisecond},
			EndOffset:        durationTestPtr(90 * time.Millisecond),
			PromptTokens:     intTestPtr(8),
		},
		{
			Index:         2,
			Status:        "canceled",
			Cancelled:     true,
			Error:         "context canceled",
			ArrivalOffset: durationTestPtr(15 * time.Millisecond),
		},
	}
	summary := Summarize(requests, 100*time.Millisecond, SLOConfig{TTFT: 30 * time.Millisecond, ITL: 15 * time.Millisecond, E2E: 80 * time.Millisecond})

	if summary.Requests != 3 || summary.Successful != 2 || summary.Cancelled != 1 || summary.Errors != 0 {
		t.Fatalf("counts = %+v", summary)
	}
	if summary.UsageRequests != 2 {
		t.Fatalf("UsageRequests = %d, want 2", summary.UsageRequests)
	}
	if !reflect.DeepEqual(summary.StatusCodes, map[int]int{200: 2}) {
		t.Fatalf("StatusCodes = %v, want map[200:2]", summary.StatusCodes)
	}
	if summary.InputTokens != 18 || summary.OutputTokens != 5 || summary.TotalTokens != 23 {
		t.Fatalf("token totals = input %d output %d total %d", summary.InputTokens, summary.OutputTokens, summary.TotalTokens)
	}
	if summary.QueueTime != (Percentiles{Count: 2, P50: 15 * time.Millisecond, P95: 19500 * time.Microsecond, P99: 19900 * time.Microsecond}) {
		t.Fatalf("queue percentiles = %+v", summary.QueueTime)
	}
	if summary.TTFT != (Percentiles{Count: 2, P50: 30 * time.Millisecond, P95: 39 * time.Millisecond, P99: 39800 * time.Microsecond}) {
		t.Fatalf("ttft percentiles = %+v", summary.TTFT)
	}
	if summary.ITL != (Percentiles{Count: 3, P50: 10 * time.Millisecond, P95: 14500 * time.Microsecond, P99: 14900 * time.Microsecond}) {
		t.Fatalf("itl percentiles = %+v", summary.ITL)
	}
	if summary.TPOT != (Percentiles{Count: 2, P50: 12500 * time.Microsecond, P95: 14750 * time.Microsecond, P99: 14950 * time.Microsecond}) {
		t.Fatalf("tpot percentiles = %+v", summary.TPOT)
	}
	if summary.E2E != (Percentiles{Count: 2, P50: 72500 * time.Microsecond, P95: 83750 * time.Microsecond, P99: 84750 * time.Microsecond}) {
		t.Fatalf("e2e percentiles = %+v", summary.E2E)
	}
	if summary.GoodRequests != 1 {
		t.Fatalf("GoodRequests = %d, want 1", summary.GoodRequests)
	}
	if summary.GoodputRatio != 1.0/3.0 {
		t.Fatalf("GoodputRatio = %v, want %v", summary.GoodputRatio, 1.0/3.0)
	}
	if summary.RequestsPerSecond != 30 || summary.GoodputPerSecond != 10 {
		t.Fatalf("throughput = requests %v goodput %v", summary.RequestsPerSecond, summary.GoodputPerSecond)
	}
	if summary.InputTokensPerSecond != 180 || summary.OutputTokensPerSecond != 50 || summary.TotalTokensPerSecond != 230 {
		t.Fatalf("token rates = %v %v %v", summary.InputTokensPerSecond, summary.OutputTokensPerSecond, summary.TotalTokensPerSecond)
	}

	report := Report{
		Config: Config{
			Endpoint:     "http://127.0.0.1:8080/v1/chat/completions",
			Model:        "test-model",
			Prompts:      []string{"one", "two"},
			RequestCount: 3,
			Concurrency:  2,
			MaxTokens:    8,
			Seed:         42,
			Timeout:      250 * time.Millisecond,
			Arrival:      ArrivalConfig{Mode: ArrivalGamma, Rate: 5, GammaShape: 2},
			SLO:          SLOConfig{TTFT: 30 * time.Millisecond},
		},
		Summary: summary,
	}
	var buf bytes.Buffer
	if err := WriteSummaryCSV(&buf, []Report{report}); err != nil {
		t.Fatalf("WriteSummaryCSV() error = %v", err)
	}
	records, err := csv.NewReader(bytes.NewReader(buf.Bytes())).ReadAll()
	if err != nil {
		t.Fatalf("csv decode: %v", err)
	}
	if len(records) != 2 {
		t.Fatalf("csv rows = %d, want 2", len(records))
	}
	row := records[1]
	if row[0] != report.Config.Endpoint || row[1] != report.Config.Model || row[5] != strconv.Itoa(report.Config.Concurrency) {
		t.Fatalf("csv row = %v", row)
	}
	if row[12] != strconv.Itoa(summary.Successful) || row[22] != strconv.Itoa(summary.TotalTokens) {
		t.Fatalf("csv summary fields = %v", row)
	}
}

func durationTestPtr(v time.Duration) *time.Duration {
	return &v
}

func intTestPtr(v int) *int {
	return &v
}
