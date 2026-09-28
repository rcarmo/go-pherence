package qwen3tts

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"
)

func TestCappedSeededCPUContextCancellationAndRecovery(t *testing.T) {
	cfg := tinyTalkerConfig()
	talker, err := LoadTalkerCPU(tinyTalkerSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	predictor, err := LoadCodePredictorCPU(tinyPredictorSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	decoder, err := LoadDecoder12HzCPU(tinyDecoderSource(tinyDecoder12HzConfig()), tinyDecoder12HzConfig())
	if err != nil {
		t.Fatal(err)
	}
	plan := tinyTalkerPlan(t, cfg)
	want, err := GenerateCappedSeededCPU(plan, talker, predictor, decoder, 42)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := GenerateCappedSeededCPUContext(nil, plan, talker, predictor, decoder, 42); err == nil || !reflect.DeepEqual(got, BoundedCPUResult{}) {
		t.Fatal("accepted nil context")
	}
	pre, cancel := context.WithCancel(context.Background())
	cancel()
	if got, err := GenerateCappedSeededCPUContext(pre, plan, talker, predictor, decoder, 42); !errors.Is(err, context.Canceled) || !reflect.DeepEqual(got, BoundedCPUResult{}) {
		t.Fatalf("pre-cancelled result=%+v err=%v", got, err)
	}
	for _, at := range []int{1, 2} {
		ctx, stop := context.WithCancel(context.Background())
		selected := 0
		got, err := generateCappedGreedyCPUContext(ctx, plan, talker, predictor, decoder, func(logits []float32, eos uint32) (uint32, error) {
			selected++
			if selected == at {
				stop()
			}
			return greedyTalkerToken(logits, eos)
		})
		stop()
		if selected != at || !errors.Is(err, context.Canceled) || !reflect.DeepEqual(got, BoundedCPUResult{}) {
			t.Fatalf("selection%d selected=%d returned partial result err=%v", at, selected, err)
		}
	}
	deadline, stop := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer stop()
	if got, err := GenerateCappedSeededCPUContext(deadline, plan, talker, predictor, decoder, 42); !errors.Is(err, context.DeadlineExceeded) || !reflect.DeepEqual(got, BoundedCPUResult{}) {
		t.Fatalf("expired result=%+v err=%v", got, err)
	}
	good, err := GenerateCappedSeededCPUContext(context.Background(), plan, talker, predictor, decoder, 42)
	if err != nil || !reflect.DeepEqual(good, want) {
		t.Fatalf("recovery mismatch err=%v", err)
	}
	// A cancelled request must not mutate shared weights or peer results.
	const workers = 8
	results := make([]BoundedCPUResult, workers)
	errs := make([]error, workers)
	var wg sync.WaitGroup
	for i := range results {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i], errs[i] = GenerateCappedSeededCPUContext(context.Background(), plan, talker, predictor, decoder, 42)
		}(i)
	}
	wg.Wait()
	for i := range results {
		if errs[i] != nil || !reflect.DeepEqual(results[i], want) {
			t.Fatalf("peer%d mismatch err=%v", i, errs[i])
		}
	}
	results[0].Waveform[0]++
	if results[0].Waveform[0] == results[1].Waveform[0] {
		t.Fatal("peer waveforms alias")
	}
}
