package pockettts

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
	"strconv"
	"sync"
	"testing"
	"time"
)

func releasedWarmSession(tb testing.TB, frames int) (*Session, NoiseSource, []float32) {
	tb.Helper()
	modelPath := releasedModel(tb)
	voicePath := releasedVoice(tb)
	gen, err := LoadGeneratorCPU(modelPath, releasedConfig(tb))
	if err != nil {
		tb.Fatal(err)
	}
	voice, err := LoadVoiceState(voicePath, gen.FlowLM.Transformer, 64)
	if err != nil {
		tb.Fatal(err)
	}
	session, err := NewSession(gen, voice, frames)
	if err != nil {
		tb.Fatal(err)
	}
	noise := func(_ int, dst []float32) error {
		for i := range dst {
			dst[i] = float32((i*7)%19-9) / 16
		}
		return nil
	}
	return session, noise, make([]float32, frames*SamplesPerFrame)
}

func TestReleasedWarmGenerateZeroAlloc(t *testing.T) {
	session, noise, pcm := releasedWarmSession(t, 5)
	tokens := []uint32{2994, 578, 682}
	if _, err := session.GenerateInto(pcm, tokens, 5, 3, 1, -4, noise); err != nil {
		t.Fatal(err)
	}
	if allocs := testing.AllocsPerRun(10, func() {
		if _, err := session.GenerateInto(pcm, tokens, 5, 3, 1, -4, noise); err != nil {
			panic(err)
		}
	}); allocs != 0 {
		t.Fatalf("warm allocations=%g", allocs)
	}
}

// Opt-in released allocation check for the uncanceled context path.
func TestReleasedWarmGenerateContextZeroAlloc(t *testing.T) {
	session, noise, pcm := releasedWarmSession(t, 5)
	tokens := []uint32{2994, 578, 682}
	ctx := context.Background()
	if _, err := session.GenerateIntoContext(ctx, pcm, tokens, 5, 3, 1, -4, noise); err != nil {
		t.Fatal(err)
	}
	if allocs := testing.AllocsPerRun(10, func() {
		if _, err := session.GenerateIntoContext(ctx, pcm, tokens, 5, 3, 1, -4, noise); err != nil {
			panic(err)
		}
	}); allocs != 0 {
		t.Fatalf("warm context allocations=%g", allocs)
	}
}

// Opt-in released cancellation check. The noise callback triggers cancellation
// at known frame/step boundaries, so the partial-output contract is testable.
func TestReleasedGenerateIntoContextCancellation(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKETTTS_CANCEL") == "" {
		t.Skip("set GO_PHERENCE_POCKETTTS_CANCEL and pinned released model/voice paths")
	}
	const frames = 5
	session, _, pcm := releasedWarmSession(t, frames)
	tokens := []uint32{2994, 578, 682}
	for _, tc := range []struct {
		name    string
		atFrame int
		want    int
	}{{"before_first_decode", 0, 0}, {"after_one_frame", 1, SamplesPerFrame}} {
		t.Run(tc.name, func(t *testing.T) {
			clear(pcm)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			noise := func(frame int, dst []float32) error {
				if frame == tc.atFrame {
					cancel()
				}
				for j := range dst {
					dst[j] = float32((j*7)%19-9) / 16
				}
				return nil
			}
			count, err := session.GenerateIntoContext(ctx, pcm, tokens, frames, 3, 2, -4, noise)
			if !errors.Is(err, context.Canceled) || count != tc.want {
				t.Fatalf("count=%d want=%d err=%v", count, tc.want, err)
			}
			for _, v := range pcm[count:] {
				if v != 0 {
					t.Fatal("cancellation wrote past completed frames")
				}
			}
			// A canceled session can be reset and reused with the legacy API.
			written, err := session.GenerateInto(pcm, tokens, frames, 3, 1, -4, func(_ int, dst []float32) error {
				for j := range dst {
					dst[j] = float32((j*7)%19-9) / 16
				}
				return nil
			})
			if err != nil || written != frames*SamplesPerFrame {
				t.Fatalf("reuse count=%d err=%v", written, err)
			}
		})
	}
}

// Opt-in asynchronous cancellation diagnostic. The noise callback signals
// another goroutine after the first decoded frame; the context is canceled
// while generation continues. Latency is measured, not asserted.
func TestReleasedGenerateIntoContextAsyncCancel(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKETTTS_CANCEL_ASYNC") == "" {
		t.Skip("set GO_PHERENCE_POCKETTTS_CANCEL_ASYNC and pinned released model/voice paths")
	}
	const frames = 5
	session, _, pcm := releasedWarmSession(t, frames)
	tokens := []uint32{2994, 578, 682}
	for run := 0; run < 5; run++ {
		clear(pcm)
		ctx, cancel := context.WithCancel(context.Background())
		requestCancel := make(chan struct{})
		cancelTime := make(chan time.Time, 1)
		go func() {
			<-requestCancel
			cancel()
			cancelTime <- time.Now()
		}()
		noise := func(frame int, dst []float32) error {
			if frame == 1 {
				close(requestCancel)
			}
			for j := range dst {
				dst[j] = float32((j*7)%19-9) / 16
			}
			return nil
		}
		count, err := session.GenerateIntoContext(ctx, pcm, tokens, frames, 3, 2, -4, noise)
		if !errors.Is(err, context.Canceled) || count%SamplesPerFrame != 0 || count < SamplesPerFrame || count >= frames*SamplesPerFrame {
			t.Fatalf("run=%d count=%d err=%v", run, count, err)
		}
		when := <-cancelTime
		for i, sample := range pcm[count:] {
			if sample != 0 {
				t.Fatalf("run=%d wrote incomplete PCM at %d", run, count+i)
			}
		}
		t.Logf("run=%d canceled_after_frame=1 completed_frames=%d signal_to_return=%s", run, count/SamplesPerFrame, time.Since(when))
	}
}

// Opt-in 25-frame comparison against a separately captured, same-host
// baseline. Reference PCM is raw little-endian F32; never hash-compare output.
// Record the reference revision, model/voice pins, architecture, and command
// with the reference file before using it as a regression oracle.
// The limits bound rounding drift, not speech quality or listening acceptance.
func TestReleasedTwentyFiveFrameNumericalParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_POCKETTTS_REFERENCE_PCM")
	if path == "" {
		t.Skip("set GO_PHERENCE_POCKETTTS_REFERENCE_PCM to independently captured 25-frame F32 baseline PCM")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	const frames = 25
	if len(data) != frames*SamplesPerFrame*4 {
		t.Fatalf("reference PCM bytes=%d want=%d", len(data), frames*SamplesPerFrame*4)
	}
	session, noise, pcm := releasedWarmSession(t, frames)
	written, err := session.GenerateInto(pcm, []uint32{2994, 578, 682}, frames, 3, 1, -4, noise)
	if err != nil {
		t.Fatal(err)
	}
	if written <= 0 || written > len(pcm) {
		t.Fatalf("invalid generated sample count %d", written)
	}
	var maxAbs, peak, sumError2, sumSignal2 float64
	var affected, aboveOneMicro int
	for i, got := range pcm {
		want := math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
		if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) {
			t.Fatalf("nonfinite PCM at sample %d: got=%g want=%g", i, got, want)
		}
		if i >= written {
			if got != 0 || want != 0 {
				t.Fatalf("nonzero output after generated sample %d at %d: got=%g want=%g", written, i, got, want)
			}
			continue
		}
		diff := math.Abs(float64(got) - float64(want))
		if diff > 0 {
			affected++
		}
		if diff > 1e-6 {
			aboveOneMicro++
		}
		maxAbs = math.Max(maxAbs, diff)
		peak = math.Max(peak, math.Abs(float64(want)))
		sumError2 += diff * diff
		sumSignal2 += float64(want) * float64(want)
	}
	if sumSignal2 == 0 {
		t.Fatal("silent reference cannot calibrate relative error")
	}
	rms := math.Sqrt(sumError2 / float64(written))
	relRMS := math.Sqrt(sumError2 / sumSignal2)
	t.Logf("25-frame-capacity PCM: generated_samples=%d buffer_samples=%d affected=%d above_1e-6=%d max_abs=%.9g rms=%.9g peak_relative=%.9g relative_rms=%.9g", written, len(pcm), affected, aboveOneMicro, maxAbs, rms, maxAbs/peak, relRMS)
	// Provisional bounds: the measured ARM64 non-fused BF16 trial had
	// max_abs=3.13e-5 and relative_rms=3.00e-5 over 28,800 emitted samples.
	// These leave rounding headroom, but are not an audio-quality admission.
	if maxAbs > 4e-5 || relRMS > 5e-5 {
		t.Fatalf("25-frame PCM drift exceeds bounds: max_abs=%.9g (limit 4e-5), relative_rms=%.9g (limit 5e-5)", maxAbs, relRMS)
	}
}

// Opt-in race/cross-talk check: request-owned sessions share immutable
// weights and voice state, but never share mutable generation buffers.
func TestReleasedConcurrentIndependentSessions(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKETTTS_CONCURRENT") == "" {
		t.Skip("set GO_PHERENCE_POCKETTTS_CONCURRENT and pinned released model/voice paths")
	}
	modelPath := releasedModel(t)
	voicePath := releasedVoice(t)
	gen, err := LoadGeneratorCPU(modelPath, releasedConfig(t))
	if err != nil {
		t.Fatal(err)
	}
	voice, err := LoadVoiceState(voicePath, gen.FlowLM.Transformer, 64)
	if err != nil {
		t.Fatal(err)
	}
	const frames = 5
	tokens := []uint32{2994, 578, 682}
	type worker struct {
		session *Session
		noise   NoiseSource
		pcm     []float32
		want    []float32
		count   int
	}
	for _, tc := range []struct {
		name    string
		workers int
		runs    int
	}{{"two_workers", 2, 3}, {"four_workers", 4, 6}} {
		t.Run(tc.name, func(t *testing.T) {
			workers := make([]worker, tc.workers)
			serial, err := NewSession(gen, voice, frames)
			if err != nil {
				t.Fatal(err)
			}
			for i := range workers {
				workers[i].session, err = NewSession(gen, voice, frames)
				if err != nil {
					t.Fatal(err)
				}
				workers[i].pcm = make([]float32, frames*SamplesPerFrame)
				workers[i].want = make([]float32, frames*SamplesPerFrame)
				shift := i * 11
				workers[i].noise = func(_ int, dst []float32) error {
					for j := range dst {
						dst[j] = float32((j*7+shift)%19-9) / 16
					}
					return nil
				}
				workers[i].count, err = serial.GenerateInto(workers[i].want, tokens, frames, 3, 1, -4, workers[i].noise)
				if err != nil {
					t.Fatal(err)
				}
			}
			for run := 0; run < tc.runs; run++ {
				var wg sync.WaitGroup
				errs := make([]error, len(workers))
				counts := make([]int, len(workers))
				start := make(chan struct{})
				for i := range workers {
					wg.Add(1)
					go func(i int) {
						defer wg.Done()
						<-start
						counts[i], errs[i] = workers[i].session.GenerateInto(workers[i].pcm, tokens, frames, 3, 1, -4, workers[i].noise)
					}(i)
				}
				close(start)
				wg.Wait()
				for i := range workers {
					if errs[i] != nil || counts[i] != workers[i].count {
						t.Fatalf("run=%d worker=%d generated=%d want=%d err=%v", run, i, counts[i], workers[i].count, errs[i])
					}
					var maxAbs, sumError2, sumSignal2 float64
					for j, got := range workers[i].pcm {
						want := workers[i].want[j]
						if !isFinitePCM(got) || !isFinitePCM(want) {
							t.Fatalf("run=%d worker=%d nonfinite PCM[%d]", run, i, j)
						}
						if j >= counts[i] {
							if got != 0 || want != 0 {
								t.Fatalf("run=%d worker=%d nonzero tail at %d", run, i, j)
							}
							continue
						}
						diff := math.Abs(float64(got) - float64(want))
						maxAbs = math.Max(maxAbs, diff)
						sumError2 += diff * diff
						sumSignal2 += float64(want) * float64(want)
					}
					if sumSignal2 == 0 {
						t.Fatalf("run=%d worker=%d silent serial reference", run, i)
					}
					relRMS := math.Sqrt(sumError2 / sumSignal2)
					if maxAbs > 4e-5 || relRMS > 5e-5 {
						t.Fatalf("run=%d worker=%d max_abs=%.9g relative_rms=%.9g", run, i, maxAbs, relRMS)
					}
					t.Logf("run=%d worker=%d generated=%d max_abs=%.9g relative_rms=%.9g", run, i, counts[i], maxAbs, relRMS)
				}
			}
		})
	}
}

// Opt-in bounded four-request timing diagnostic. Preparation and serial
// baselines are outside the timer; results are logged, not timing assertions.
func TestReleasedFourSessionTiming(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKETTTS_CONCURRENT_TIMING") == "" {
		t.Skip("set GO_PHERENCE_POCKETTTS_CONCURRENT_TIMING and pinned released model/voice paths")
	}
	modelPath := releasedModel(t)
	voicePath := releasedVoice(t)
	gen, err := LoadGeneratorCPU(modelPath, releasedConfig(t))
	if err != nil {
		t.Fatal(err)
	}
	voice, err := LoadVoiceState(voicePath, gen.FlowLM.Transformer, 64)
	if err != nil {
		t.Fatal(err)
	}
	const (
		workers = 4
		frames  = 5
		rounds  = 5
	)
	tokens := []uint32{2994, 578, 682}
	sessions := make([]*Session, workers)
	outputs := make([][]float32, workers)
	noise := func(_ int, dst []float32) error {
		for j := range dst {
			dst[j] = float32((j*7)%19-9) / 16
		}
		return nil
	}
	for i := range sessions {
		sessions[i], err = NewSession(gen, voice, frames)
		if err != nil {
			t.Fatal(err)
		}
		outputs[i] = make([]float32, frames*SamplesPerFrame)
		if _, err := sessions[i].GenerateInto(outputs[i], tokens, frames, 3, 1, -4, noise); err != nil {
			t.Fatal(err)
		}
	}
	// Serial and parallel runs use the same warmed sessions and output buffers.
	for _, tc := range []struct {
		name     string
		parallel bool
	}{{"serial", false}, {"parallel", true}} {
		for trial := 0; trial < 3; trial++ {
			errs := make([]error, workers)
			begin := time.Now()
			for round := 0; round < rounds; round++ {
				if !tc.parallel {
					for i := range sessions {
						if _, err := sessions[i].GenerateInto(outputs[i], tokens, frames, 3, 1, -4, noise); err != nil {
							t.Fatal(err)
						}
					}
					continue
				}
				var wg sync.WaitGroup
				for i := range sessions {
					wg.Add(1)
					go func(i int) {
						defer wg.Done()
						_, errs[i] = sessions[i].GenerateInto(outputs[i], tokens, frames, 3, 1, -4, noise)
					}(i)
				}
				wg.Wait()
				for _, err := range errs {
					if err != nil {
						t.Fatal(err)
					}
				}
			}
			t.Logf("%s trial=%d total_requests=%d elapsed=%s", tc.name, trial, rounds*workers, time.Since(begin))
		}
	}
}

// Opt-in bounded shared-model reuse check. It measures request latency under
// four concurrent workers, but is not a production soak or quality judgement.
func TestReleasedFourSessionReuse(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKETTTS_REUSE") == "" {
		t.Skip("set GO_PHERENCE_POCKETTTS_REUSE and pinned released model/voice paths")
	}
	modelPath := releasedModel(t)
	voicePath := releasedVoice(t)
	gen, err := LoadGeneratorCPU(modelPath, releasedConfig(t))
	if err != nil {
		t.Fatal(err)
	}
	voice, err := LoadVoiceState(voicePath, gen.FlowLM.Transformer, 64)
	if err != nil {
		t.Fatal(err)
	}
	const (
		workers = 4
		frames  = 5
	)
	requests := 40
	if value := os.Getenv("GO_PHERENCE_POCKETTTS_REUSE_REQUESTS"); value != "" {
		requests, err = strconv.Atoi(value)
		if err != nil || requests < 1 || requests > 200 {
			t.Fatalf("GO_PHERENCE_POCKETTTS_REUSE_REQUESTS must be 1..200: %q", value)
		}
	}
	tokens := []uint32{2994, 578, 682}
	type work struct {
		session   *Session
		pcm, want []float32
		noise     NoiseSource
		count     int
		durations []time.Duration
		err       error
	}
	var jobs [workers]work
	serial, err := NewSession(gen, voice, frames)
	if err != nil {
		t.Fatal(err)
	}
	for i := range jobs {
		job := &jobs[i]
		job.session, err = NewSession(gen, voice, frames)
		if err != nil {
			t.Fatal(err)
		}
		job.pcm = make([]float32, frames*SamplesPerFrame)
		job.want = make([]float32, frames*SamplesPerFrame)
		job.durations = make([]time.Duration, requests)
		shift := i * 11
		job.noise = func(_ int, dst []float32) error {
			for j := range dst {
				dst[j] = float32((j*7+shift)%19-9) / 16
			}
			return nil
		}
		job.count, err = serial.GenerateInto(job.want, tokens, frames, 3, 1, -4, job.noise)
		if err != nil || job.count <= 0 {
			t.Fatalf("serial worker=%d count=%d err=%v", i, job.count, err)
		}
		if _, err := job.session.GenerateInto(job.pcm, tokens, frames, 3, 1, -4, job.noise); err != nil {
			t.Fatal(err)
		}
	}
	// Retained heap is reported after a GC outside the measured request window.
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	start := make(chan struct{})
	var wg sync.WaitGroup
	begin := time.Now()
	for i := range jobs {
		wg.Add(1)
		go func(job *work) {
			defer wg.Done()
			<-start
			for run := 0; run < requests; run++ {
				requestStart := time.Now()
				count, err := job.session.GenerateInto(job.pcm, tokens, frames, 3, 1, -4, job.noise)
				job.durations[run] = time.Since(requestStart)
				if err != nil || count != job.count {
					job.err = fmt.Errorf("request=%d count=%d want=%d: %v", run, count, job.count, err)
					return
				}
				var maxAbs, sumError2, sumSignal2 float64
				for j, got := range job.pcm {
					want := job.want[j]
					if !isFinitePCM(got) || !isFinitePCM(want) || (j >= count && (got != 0 || want != 0)) {
						job.err = fmt.Errorf("request=%d invalid PCM[%d]", run, j)
						return
					}
					if j >= count {
						continue
					}
					diff := math.Abs(float64(got) - float64(want))
					maxAbs = math.Max(maxAbs, diff)
					sumError2 += diff * diff
					sumSignal2 += float64(want) * float64(want)
				}
				if sumSignal2 == 0 || maxAbs > 4e-5 || math.Sqrt(sumError2/sumSignal2) > 5e-5 {
					job.err = fmt.Errorf("request=%d max_abs=%.9g relative_rms=%.9g", run, maxAbs, math.Sqrt(sumError2/sumSignal2))
					return
				}
			}
		}(&jobs[i])
	}
	close(start)
	wg.Wait()
	elapsed := time.Since(begin)
	for i := range jobs {
		if jobs[i].err != nil {
			t.Fatalf("worker=%d: %v", i, jobs[i].err)
		}
	}
	runtime.GC()
	runtime.ReadMemStats(&after)
	latencies := make([]time.Duration, 0, workers*requests)
	for i := range jobs {
		latencies = append(latencies, jobs[i].durations...)
	}
	slices.Sort(latencies)
	t.Logf("four-worker reuse: requests=%d samples_each=%d elapsed=%s latency_min=%s p50=%s p95=%s p99=%s max=%s heap_after_GC_before=%d after=%d bytes", len(latencies), jobs[0].count, elapsed, latencies[0], latencies[len(latencies)/2], latencies[(len(latencies)*95+99)/100-1], latencies[(len(latencies)*99+99)/100-1], latencies[len(latencies)-1], before.HeapAlloc, after.HeapAlloc)
}

func isFinitePCM(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

func benchmarkReleasedFrames(b *testing.B, frames int) {
	session, noise, pcm := releasedWarmSession(b, frames)
	tokens := []uint32{2994, 578, 682}
	written, err := session.GenerateInto(pcm, tokens, frames, 3, 1, -4, noise)
	if err != nil {
		b.Fatal(err)
	}
	if written <= 0 || written > len(pcm) {
		b.Fatalf("invalid generated sample count %d", written)
	}
	b.ReportAllocs()
	b.SetBytes(int64(written * 4))
	b.ResetTimer()
	for b.Loop() {
		count, err := session.GenerateInto(pcm, tokens, frames, 3, 1, -4, noise)
		if err != nil || count != written {
			b.Fatalf("generated %d samples, want %d: %v", count, written, err)
		}
	}
}
func BenchmarkReleasedFirstChunk(b *testing.B)       { benchmarkReleasedFrames(b, 1) }
func BenchmarkReleasedFiveFrames(b *testing.B)       { benchmarkReleasedFrames(b, 5) }
func BenchmarkReleasedTwentyFiveFrames(b *testing.B) { benchmarkReleasedFrames(b, 25) }
