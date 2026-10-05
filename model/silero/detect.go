package silero

import (
	"context"
	"fmt"
	"io"
)

// SampleReader supplies original-timeline mono16-kHz PCM, without retaining
// caller scratch. media.PCMReader and Whisper's SampleReader satisfy it.
type SampleReader interface {
	ReadSamplesAt(context.Context, []float32, int64) (int, error)
}

// Probabilities starts a fresh recurrent stream and returns one owned score per
// 512-sample window. It reads bounded64K-sample batches; EOF padding occurs only
// after the actual tail. A failure returns no partial scores. Scores and model
// state are never checkpointed as a successful interrupted VAD result.
func (m *Model) Probabilities(ctx context.Context, source SampleReader, totalSamples int64) ([]float32, error) {
	if ctx == nil || source == nil || totalSamples < 1 || totalSamples > 4*3600*16000 {
		return nil, fmt.Errorf("Silero: invalid PCM request")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	stream, err := m.NewStream()
	if err != nil {
		return nil, err
	}
	scores := make([]float32, 0, (totalSamples+511)/512)
	scratch := make([]float32, 65536)
	for offset := int64(0); offset < totalSamples; {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		count := int(min(int64(len(scratch)), totalSamples-offset))
		n, err := source.ReadSamplesAt(ctx, scratch[:count], offset)
		if err != nil && err != io.EOF {
			return nil, err
		}
		if n != count {
			return nil, io.ErrUnexpectedEOF
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for start := 0; start < count; start += 512 {
			end := min(start+512, count)
			var tail [512]float32
			frame := scratch[start:end]
			if end-start != 512 {
				copy(tail[:], frame)
				frame = tail[:]
			}
			score, err := stream.Probability(ctx, frame)
			if err != nil {
				return nil, err
			}
			scores = append(scores, score)
		}
		offset += int64(count)
	}
	return scores, nil
}

// Detect returns original sample spans. It owns all inference state, independent
// of every other call; no GPU, external process, energy-threshold substitute or
// silent inference fallback is used.
func (m *Model) Detect(ctx context.Context, source SampleReader, totalSamples int64, opts SegmentOptions) ([]Span, error) {
	if err := opts.Validate(); err != nil {
		return nil, err
	}
	scores, err := m.Probabilities(ctx, source, totalSamples)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return SpeechSpans(scores, totalSamples, opts)
}

// CompactedReader supplies only retained speech. It maps each bounded read to
// the original source, including reads crossing gaps. It owns metadata, never
// PCM, and is safe only under the source reader's concurrency contract.
type CompactedReader struct {
	source  SampleReader
	mapping *TimeMap
}

func NewCompactedReader(source SampleReader, spans []Span, totalSamples int64) (*CompactedReader, error) {
	if source == nil {
		return nil, fmt.Errorf("Silero: nil source")
	}
	mapping, err := NewTimeMap(spans, totalSamples)
	if err != nil {
		return nil, err
	}
	return &CompactedReader{source: source, mapping: mapping}, nil
}
func (r *CompactedReader) Samples() int64 {
	if r == nil {
		return 0
	}
	return r.mapping.Samples()
}
func (r *CompactedReader) MapStart(sample int64) (int64, error) {
	if r == nil {
		return 0, fmt.Errorf("Silero: nil reader")
	}
	return r.mapping.MapStart(sample)
}
func (r *CompactedReader) MapEnd(sample int64) (int64, error) {
	if r == nil {
		return 0, fmt.Errorf("Silero: nil reader")
	}
	return r.mapping.MapEnd(sample)
}
func (r *CompactedReader) ReadSamplesAt(ctx context.Context, dst []float32, start int64) (int, error) {
	if ctx == nil || r == nil || r.source == nil || r.mapping == nil || start < 0 || start > r.mapping.total {
		return 0, fmt.Errorf("Silero: invalid compacted read")
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if len(dst) == 0 {
		return 0, nil
	}
	if start == r.mapping.total {
		return 0, io.EOF
	}
	m := r.mapping
	left, right := 0, len(m.spans)
	for left < right {
		mid := (left + right) / 2
		if m.offsets[mid]+m.spans[mid].End-m.spans[mid].Start <= start {
			left = mid + 1
		} else {
			right = mid
		}
	}
	written := 0
	for i := left; i < len(m.spans) && written < len(dst); i++ {
		if err := ctx.Err(); err != nil {
			return written, err
		}
		original := m.spans[i].Start + start - m.offsets[i]
		count := int(min(int64(len(dst)-written), m.spans[i].End-original))
		n, err := r.source.ReadSamplesAt(ctx, dst[written:written+count], original)
		if n < 0 || n > count {
			return written, fmt.Errorf("Silero: invalid reader sample count")
		}
		written += n
		start += int64(n)
		if err != nil && err != io.EOF {
			return written, err
		}
		if n != count {
			return written, io.ErrUnexpectedEOF
		}
		if err := ctx.Err(); err != nil {
			return written, err
		}
	}
	if written < len(dst) {
		return written, io.EOF
	}
	return written, nil
}
