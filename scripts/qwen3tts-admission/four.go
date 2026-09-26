//go:build linux

package main

import (
	"fmt"
	"reflect"
	"runtime"
	"sync"
	"time"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model/qwen3tts"
)

// probeFourMixed shares immutable weights across four capped requests. The
// fixtures are independently pinned by main before this path is entered.
func probeFourMixed(plans [2]qwen3tts.RuntimeRequestPlan, fixtures [2]fixture,
	t *qwen3tts.TalkerCPU, p *qwen3tts.CodePredictorCPU,
	d *qwen3tts.Decoder12HzCPU, tok *tokenizer.Tokenizer, base uint64) {
	const callers = 4
	invalid := plans[0]
	invalid.MaxFrames = qwen3tts.MaxCappedCPUFrames + 1
	if partial, err := qwen3tts.GenerateCappedSeededCPU(invalid, t, p, d, fixtures[0].seed); err == nil || !reflect.DeepEqual(partial, qwen3tts.BoundedCPUResult{}) {
		panic("invalid capped request returned partial output")
	}
	fmt.Println("invalid_request_rejected_without_partial=true")
	var results [callers]qwen3tts.BoundedCPUResult
	var errs [callers]error
	var took [callers]time.Duration
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := range results {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			begin := time.Now()
			f := fixtures[i%2]
			results[i], errs[i] = qwen3tts.GenerateCappedSeededCPU(plans[i%2], t, p, d, f.seed)
			took[i] = time.Since(begin)
		}(i)
	}
	begin := time.Now()
	close(start)
	wg.Wait()
	fmt.Printf("four_returned wall=%s\n", time.Since(begin))
	inspect("four_returned")
	for i, r := range results {
		if errs[i] != nil {
			panic(fmt.Sprintf("request_%d: %v", i, errs[i]))
		}
		f := fixtures[i%2]
		fmt.Printf("request_%d text=%q seed=%d took=%s max_abs=%g\n", i, f.text, f.seed, took[i], check(r, f.codes, f.wave, f.frames, f.threshold))
		for j := 0; j < i; j++ {
			other := results[j]
			if &r.Semantic[0] == &other.Semantic[0] || &r.Acoustic[0] == &other.Acoustic[0] || &r.Waveform[0] == &other.Waveform[0] {
				panic(fmt.Sprintf("results %d/%d alias", i, j))
			}
		}
	}
	// Mutate one owned output, then verify every peer still matches its oracle.
	results[0].Waveform[0]++
	for i := 1; i < callers; i++ {
		f := fixtures[i%2]
		check(results[i], f.codes, f.wave, f.frames, f.threshold)
	}
	fmt.Println("independent_owned_results=true")
	results = [callers]qwen3tts.BoundedCPUResult{}
	runtime.GC()
	after, peak := inspect("after_four_post_gc")
	if after > base+1048576 {
		panic(fmt.Sprintf("retained heap grew by %d bytes", after-base))
	}
	runtime.KeepAlive(t)
	runtime.KeepAlive(p)
	runtime.KeepAlive(d)
	runtime.KeepAlive(tok)
	runtime.KeepAlive(plans)
	fmt.Printf("post_gc_delta_bytes=%d peak_rss_kib=%d\n", int64(after)-int64(base), peak)
}
