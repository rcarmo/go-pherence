package silero

import (
	"context"
	"errors"
	"io"
	"reflect"
	"testing"
)

type memoryPCM struct {
	data         []float32
	short        bool
	fail         error
	starts       []int64
	cancel       context.CancelFunc
	invalidCount bool
}

func (r *memoryPCM) ReadSamplesAt(ctx context.Context, dst []float32, start int64) (int, error) {
	r.starts = append(r.starts, start)
	if r.fail != nil {
		return 0, r.fail
	}
	if r.invalidCount {
		return len(dst) + 1, nil
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	count := copy(dst, r.data[start:])
	if r.cancel != nil {
		r.cancel()
	}
	if r.short && count > 0 {
		return count - 1, io.EOF
	}
	if count < len(dst) {
		return count, io.EOF
	}
	return count, nil
}
func TestProbabilitiesFreshStreamAndTail(t *testing.T) {
	model := mustModel(t)
	source := &memoryPCM{data: make([]float32, 1200)}
	source.data[0] = 0.25
	before := append([]float32(nil), source.data...)
	got, err := model.Probabilities(context.Background(), source, 1200)
	if err != nil || len(got) != 3 {
		t.Fatal(got, err)
	}
	stream, _ := model.NewStream()
	var reference []float32
	for offset := 0; offset < 1200; offset += 512 {
		var frame [512]float32
		copy(frame[:], source.data[offset:min(offset+512, 1200)])
		p, err := stream.Probability(context.Background(), frame[:])
		if err != nil {
			t.Fatal(err)
		}
		reference = append(reference, p)
	}
	if !reflect.DeepEqual(got, reference) || !reflect.DeepEqual(source.data, before) {
		t.Fatal("tail/preservation")
	}
	fresh, err := model.Probabilities(context.Background(), source, 1200)
	if err != nil || !reflect.DeepEqual(fresh, got) {
		t.Fatal("inherited state")
	}
	spans, err := model.Detect(context.Background(), source, 1200, SegmentOptions{Threshold: 0.5, MinSilenceSamples: 1})
	if err != nil || !reflect.DeepEqual(spans, []Span{{0, 1200}}) {
		t.Fatal("detection", spans, err)
	}
}
func TestProbabilitiesReaderAndCancellationFailures(t *testing.T) {
	m := mustModel(t)
	source := &memoryPCM{data: make([]float32, 512)}
	if _, err := m.Probabilities(nil, source, 512); err == nil {
		t.Fatal("nil context")
	}
	if _, err := m.Probabilities(context.Background(), nil, 512); err == nil {
		t.Fatal("nil source")
	}
	for _, total := range []int64{0, 4*3600*16000 + 1} {
		if _, err := m.Probabilities(context.Background(), source, total); err == nil {
			t.Fatal("invalid total")
		}
	}
	if _, err := (*Model)(nil).Probabilities(context.Background(), source, 512); err == nil {
		t.Fatal("nil model")
	}
	sentinel := errors.New("reader failed")
	source.fail = sentinel
	if got, err := m.Probabilities(context.Background(), source, 512); !errors.Is(err, sentinel) || got != nil {
		t.Fatal("read failure", got, err)
	}
	source.fail = nil
	source.short = true
	if got, err := m.Probabilities(context.Background(), source, 512); !errors.Is(err, io.ErrUnexpectedEOF) || got != nil {
		t.Fatal("short read", got, err)
	}
	source.short = false
	ctx, cancel := context.WithCancel(context.Background())
	source.cancel = cancel
	if got, err := m.Probabilities(ctx, source, 512); !errors.Is(err, context.Canceled) || got != nil {
		t.Fatal("partial cancelled scores", got, err)
	}
	if _, err := m.Detect(ctx, source, 512, DefaultSegmentOptions()); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
func TestCompactedReaderMapsAcrossGapsWithoutRetainingPCM(t *testing.T) {
	source := &memoryPCM{data: make([]float32, 60)}
	for i := range source.data {
		source.data[i] = float32(i)
	}
	spans := []Span{{5, 10}, {20, 25}, {50, 53}}
	r, err := NewCompactedReader(source, spans, 60)
	if err != nil {
		t.Fatal(err)
	}
	spans[0].Start = 0
	if r.Samples() != 13 {
		t.Fatal("length")
	}
	got := make([]float32, 10)
	n, err := r.ReadSamplesAt(context.Background(), got, 3)
	want := []float32{8, 9, 20, 21, 22, 23, 24, 50, 51, 52}
	if err != nil || n != 10 || !reflect.DeepEqual(got, want) {
		t.Fatal("gap read", got, n, err)
	}
	if !reflect.DeepEqual(source.starts, []int64{8, 20, 50}) {
		t.Fatal("original reads", source.starts)
	}
	got = make([]float32, 5)
	n, err = r.ReadSamplesAt(context.Background(), got, 11)
	if !errors.Is(err, io.EOF) || n != 2 || got[0] != 51 || got[1] != 52 {
		t.Fatal("EOF", got, n, err)
	}
	if x, err := r.MapStart(5); err != nil || x != 20 {
		t.Fatal(x, err)
	}
	if x, err := r.MapEnd(5); err != nil || x != 10 {
		t.Fatal(x, err)
	}
	if _, err := r.ReadSamplesAt(context.Background(), nil, 13); err != nil {
		t.Fatal("empty EOF")
	}
	if n, err := r.ReadSamplesAt(context.Background(), make([]float32, 1), 13); n != 0 || !errors.Is(err, io.EOF) {
		t.Fatal(n, err)
	}
}
func TestCompactedReaderRejectsAndPropagatesFailures(t *testing.T) {
	source := &memoryPCM{data: make([]float32, 100)}
	if _, err := NewCompactedReader(nil, nil, 100); err == nil {
		t.Fatal("nil source")
	}
	if _, err := NewCompactedReader(source, []Span{{10, 10}}, 100); err == nil {
		t.Fatal("bad spans")
	}
	r, _ := NewCompactedReader(source, []Span{{0, 10}}, 100)
	if _, err := r.ReadSamplesAt(nil, make([]float32, 1), 0); err == nil {
		t.Fatal("nil context")
	}
	for _, start := range []int64{-1, 11} {
		if _, err := r.ReadSamplesAt(context.Background(), make([]float32, 1), start); err == nil {
			t.Fatal("bad start")
		}
	}
	if _, err := (*CompactedReader)(nil).ReadSamplesAt(context.Background(), nil, 0); err == nil {
		t.Fatal("nil reader")
	}
	if (*CompactedReader)(nil).Samples() != 0 {
		t.Fatal("nil length")
	}
	if _, err := (*CompactedReader)(nil).MapStart(0); err == nil {
		t.Fatal("nil start")
	}
	if _, err := (*CompactedReader)(nil).MapEnd(0); err == nil {
		t.Fatal("nil end")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := r.ReadSamplesAt(ctx, make([]float32, 1), 0); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	sentinel := errors.New("reader")
	source.fail = sentinel
	if _, err := r.ReadSamplesAt(context.Background(), make([]float32, 1), 0); !errors.Is(err, sentinel) {
		t.Fatal(err)
	}
	source.fail = nil
	source.invalidCount = true
	if _, err := r.ReadSamplesAt(context.Background(), make([]float32, 1), 0); err == nil {
		t.Fatal("invalid source count")
	}
	source.invalidCount = false
	source.short = true
	if _, err := r.ReadSamplesAt(context.Background(), make([]float32, 1), 0); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatal(err)
	}
}

func TestCompactedReaderFinalReadObservesCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	source := &memoryPCM{data: make([]float32, 10), cancel: cancel}
	reader, err := NewCompactedReader(source, []Span{{Start: 0, End: 10}}, 10)
	if err != nil {
		t.Fatal(err)
	}
	n, err := reader.ReadSamplesAt(ctx, make([]float32, 10), 0)
	if n != 10 || !errors.Is(err, context.Canceled) {
		t.Fatal("cancelled final read returned success", n, err)
	}
}
