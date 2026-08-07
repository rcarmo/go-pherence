package model

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/runtime/inferencesched"
)

type gemmaSchedRequest struct {
	id  string
	seq uint64
}

func (r gemmaSchedRequest) ID() string         { return r.id }
func (r gemmaSchedRequest) ArrivalSeq() uint64 { return r.seq }

func BenchmarkGemma4SchedulerInterferenceRealE4B(b *testing.B) {
	if os.Getenv("GO_PHERENCE_GEMMA4_SCHED_REAL") == "" {
		b.Skip("set GO_PHERENCE_GEMMA4_SCHED_REAL=1")
	}
	path := os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	if path == "" {
		path = filepath.Join(findMTPGraphBenchRepoRoot(), "models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf")
	}
	m, err := LoadGemma4GGUFAsLlama(path)
	if err != nil {
		b.Fatal(err)
	}
	activePrompt := []int{10979}
	longPrompt := []int{10979, 236764, 10979, 236764, 10979, 236764, 10979}
	shortPrompt := []int{236764}
	for _, quantum := range []int{1, 2, 4, 64} {
		b.Run(fmt.Sprintf("quantum_%d", quantum), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				activeS, _ := NewGemma4DecodeSession(m, SessionOptions{Backend: InferenceBackendSIMD, MaxTokens: 3})
				longS, _ := NewGemma4DecodeSession(m, SessionOptions{Backend: InferenceBackendSIMD, MaxTokens: 1})
				shortS, _ := NewGemma4DecodeSession(m, SessionOptions{Backend: InferenceBackendSIMD, MaxTokens: 1})
				active, _ := NewGemma4ScheduledWork(activeS, activePrompt)
				long, _ := NewGemma4ScheduledWork(longS, longPrompt)
				short, _ := NewGemma4ScheduledWork(shortS, shortPrompt)
				s, _ := inferencesched.New(inferencesched.Config{MaxActive: 3, DecodeBudget: 1, PrefillBudget: quantum})
				_ = s.Add(context.Background(), gemmaSchedRequest{"active", 1}, active)
				_ = s.Add(context.Background(), gemmaSchedRequest{"long", 2}, long)
				_ = s.Add(context.Background(), gemmaSchedRequest{"short", 3}, short)
				b.StartTimer()
				start := time.Now()
				var activeTimes []time.Time
				var shortFirst time.Duration
				for step := 0; step < 64; step++ {
					r, err := s.Step(context.Background())
					if err != nil {
						b.Fatal(err)
					}
					if got := active.DrainResults(); len(got) > 0 {
						for range got {
							activeTimes = append(activeTimes, time.Now())
						}
					}
					if got := short.DrainResults(); len(got) > 0 && shortFirst == 0 {
						shortFirst = time.Since(start)
					}
					if r.Stats.Running == 0 && r.Stats.Waiting == 0 {
						break
					}
				}
				elapsed := time.Since(start)
				b.StopTimer()
				_ = s.Close()
				if len(activeTimes) > 1 {
					b.ReportMetric(float64(activeTimes[1].Sub(activeTimes[0]))/float64(time.Millisecond), "active_itl_ms")
				}
				b.ReportMetric(float64(shortFirst)/float64(time.Millisecond), "short_ttft_ms")
				b.ReportMetric(float64(elapsed)/float64(time.Millisecond), "total_ms")
			}
		})
	}
}
