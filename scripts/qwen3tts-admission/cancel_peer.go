//go:build linux

package main

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model/qwen3tts"
)

// cancelAfterPrefill has no timer. It observes the context check reached after
// Prefill and cancels there; all state belongs to this one request.
type cancelAfterPrefill struct {
	context.Context
	mu     sync.Mutex
	checks int
	cancel context.CancelFunc
}

func (c *cancelAfterPrefill) Err() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.checks++
	if c.checks == 2 {
		c.cancel()
	}
	return c.Context.Err()
}

func (c *cancelAfterPrefill) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.checks
}

func probeCancelledPeer(plans [2]qwen3tts.RuntimeRequestPlan, fixtures [2]fixture,
	t *qwen3tts.TalkerCPU, p *qwen3tts.CodePredictorCPU,
	d *qwen3tts.Decoder12HzCPU, tok *tokenizer.Tokenizer, base uint64) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	marked := &cancelAfterPrefill{Context: ctx, cancel: cancel}
	var cancelled, peer qwen3tts.BoundedCPUResult
	var cancelErr, peerErr error
	var wg sync.WaitGroup
	start := make(chan struct{})
	wg.Add(2)
	go func() {
		defer wg.Done()
		<-start
		cancelled, cancelErr = qwen3tts.GenerateCappedSeededCPUContext(marked, plans[0], t, p, d, fixtures[0].seed)
	}()
	go func() {
		defer wg.Done()
		<-start
		peer, peerErr = qwen3tts.GenerateCappedSeededCPU(plans[1], t, p, d, fixtures[1].seed)
	}()
	begin := time.Now()
	close(start)
	wg.Wait()
	fmt.Printf("cancel_peer_returned wall=%s context_checks=%d\n", time.Since(begin), marked.count())
	inspect("cancel_peer_returned")
	if marked.count() != 2 || !errors.Is(cancelErr, context.Canceled) || len(cancelled.Semantic) != 0 || len(cancelled.Acoustic) != 0 || len(cancelled.Waveform) != 0 || len(cancelled.ContinuationHidden) != 0 || len(cancelled.ContinuationLogits) != 0 {
		panic(fmt.Sprintf("cancelled request escaped partial data: err=%v", cancelErr))
	}
	fmt.Println("post_prefill_cancellation_empty=true")
	if peerErr != nil {
		panic(peerErr)
	}
	fmt.Printf("peer_max_abs=%g\n", check(peer, fixtures[1].codes, fixtures[1].wave, fixtures[1].frames, fixtures[1].threshold))
	recovered, err := qwen3tts.GenerateCappedSeededCPUContext(context.Background(), plans[0], t, p, d, fixtures[0].seed)
	if err != nil {
		panic(err)
	}
	fmt.Printf("recovery_max_abs=%g\n", check(recovered, fixtures[0].codes, fixtures[0].wave, fixtures[0].frames, fixtures[0].threshold))
	if &peer.Semantic[0] == &recovered.Semantic[0] || &peer.Acoustic[0] == &recovered.Acoustic[0] || &peer.Waveform[0] == &recovered.Waveform[0] {
		panic("peer/recovery outputs alias")
	}
	peer.Waveform[0]++
	check(recovered, fixtures[0].codes, fixtures[0].wave, fixtures[0].frames, fixtures[0].threshold)
	fmt.Println("independent_owned_peer_recovery=true")
	peer, recovered = qwen3tts.BoundedCPUResult{}, qwen3tts.BoundedCPUResult{}
	runtime.GC()
	after, peak := inspect("cancel_peer_post_gc")
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
