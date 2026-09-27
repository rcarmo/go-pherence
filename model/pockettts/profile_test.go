package pockettts

import (
	"encoding/binary"
	"math"
	"os"
	"sync"
	"testing"
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

func isFinitePCM(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

func benchmarkReleasedFrames(b *testing.B, frames int) {
	session, noise, pcm := releasedWarmSession(b, frames)
	tokens := []uint32{2994, 578, 682}
	b.ReportAllocs()
	b.SetBytes(int64(frames * SamplesPerFrame * 4))
	b.ResetTimer()
	for b.Loop() {
		if _, err := session.GenerateInto(pcm, tokens, frames, 3, 1, -4, noise); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkReleasedFirstChunk(b *testing.B)       { benchmarkReleasedFrames(b, 1) }
func BenchmarkReleasedFiveFrames(b *testing.B)       { benchmarkReleasedFrames(b, 5) }
func BenchmarkReleasedTwentyFiveFrames(b *testing.B) { benchmarkReleasedFrames(b, 25) }
