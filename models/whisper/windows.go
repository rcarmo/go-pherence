package whisper

import (
	"context"
	"fmt"
	"io"
)

const (
	SpeechSampleRate int64 = 16000
	MaxWindowSamples int64 = 30 * SpeechSampleRate
	MinWindowSamples int64 = 160
)

// WindowPlan describes contiguous, overlapping windows without allocating a
// list or holding the recording. It is immutable, safe to share and has no
// historical 100-chunk cutoff. Input counts refer to canonical 16 kHz PCM.
// Window geometry is explicit: validation of a model's required padding and
// maximum recording/job budget remains the caller's responsibility.
type WindowPlan struct {
	total   int64
	length  int64
	overlap int64
	count   int64
}

// Window is one analysis span [Start, End) with a disjoint output ownership
// interval [EmitStart, EmitEnd). Adjacent ownership intervals meet at the middle
// of their shared context (odd overlap gives the earlier window one extra
// sample). The last window is right-padded with zeros to InputSamples.
//
// These are eligibility intervals for future word/timestamp merging, not an
// implementation of transcript deduplication or a speaker timeline. The job
// layer must map canonical PCM offsets to the original container's timebase.
type Window struct {
	Index        int64
	Start        int64
	End          int64
	EmitStart    int64
	EmitEnd      int64
	InputSamples int64
	PadSamples   int64
}

// NewWindowPlan accepts empty recordings (zero windows), bounds an individual
// analysis window to 10 ms..30 s and rejects overlaps that prevent progress.
// Total and Count are int64; arithmetic does not round via floating-point seconds.
// It does not cap a recording silently. Applications must enforce their job
// duration/window-count limit before scheduling any work.
func NewWindowPlan(totalSamples, windowSamples, overlapSamples int64) (WindowPlan, error) {
	if totalSamples < 0 || windowSamples < MinWindowSamples || windowSamples > MaxWindowSamples || overlapSamples < 0 || overlapSamples >= windowSamples {
		return WindowPlan{}, fmt.Errorf("invalid Whisper window geometry")
	}
	p := WindowPlan{total: totalSamples, length: windowSamples, overlap: overlapSamples}
	if totalSamples == 0 {
		return p, nil
	}
	p.count = 1
	if totalSamples > windowSamples {
		step := windowSamples - overlapSamples
		// ceil((total-length)/step), without adding step-1 to a large total.
		p.count += 1 + (totalSamples-windowSamples-1)/step
	}
	return p, nil
}

func (p WindowPlan) Count() int64        { return p.count }
func (p WindowPlan) TotalSamples() int64 { return p.total }

// At computes one window on demand. It rejects a zero/uninitialised plan and
// indexes outside [0, Count), including all indexes for an empty recording.
func (p WindowPlan) At(index int64) (Window, error) {
	if p.length < MinWindowSamples || index < 0 || index >= p.count {
		return Window{}, fmt.Errorf("Whisper window index out of range")
	}
	// index<count guarantees start<total without a multiplication overflow.
	start := index * (p.length - p.overlap)
	valid := p.length
	if valid > p.total-start {
		valid = p.total - start
	}
	w := Window{Index: index, Start: start, End: start + valid, InputSamples: p.length, PadSamples: p.length - valid}
	w.EmitStart = w.Start
	if index > 0 {
		w.EmitStart += p.overlap - p.overlap/2
	}
	w.EmitEnd = w.End
	if index < p.count-1 {
		w.EmitEnd -= p.overlap / 2
	}
	return w, nil
}

// SampleReader is the consumer-side PCM boundary. media.PCMReader implements it;
// a later go-264 adapter can implement it without importing Whisper. Implementors
// must honour context cancellation and treat an unexpected short read as error.
type SampleReader interface {
	ReadSamplesAt(ctx context.Context, dst []float32, startSample int64) (int, error)
}

// ReadWindow loads one window into caller-owned scratch, pads only its missing
// right tail, and leaves dst[InputSamples:] untouched. The buffer may be reused
// for every window; no recording-size or window-count allocation is performed.
// On error callers must discard the partial window. This does not compute model
// features or launch an encoder/decoder.
func (p WindowPlan) ReadWindow(ctx context.Context, source SampleReader, index int64, dst []float32) (Window, error) {
	if err := ctx.Err(); err != nil {
		return Window{}, err
	}
	w, err := p.At(index)
	if err != nil {
		return Window{}, err
	}
	if source == nil || int64(len(dst)) < w.InputSamples {
		return Window{}, fmt.Errorf("Whisper window requires source and adequate scratch")
	}
	valid := int(w.End - w.Start)
	n, readErr := source.ReadSamplesAt(ctx, dst[:valid], w.Start)
	if err := ctx.Err(); err != nil {
		return Window{}, err
	}
	if n != valid || (readErr != nil && readErr != io.EOF) {
		if readErr == nil || readErr == io.EOF {
			readErr = io.ErrUnexpectedEOF
		}
		return Window{}, fmt.Errorf("read Whisper window %d: %w", index, readErr)
	}
	clear(dst[valid:int(w.InputSamples)])
	return w, nil
}
