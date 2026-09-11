package whisper

import (
	"context"
	"encoding/binary"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio/media"
)

func checkWindowCoverage(t *testing.T, total, length, overlap int64) {
	t.Helper()
	p, err := NewWindowPlan(total, length, overlap)
	if err != nil {
		t.Fatal(err)
	}
	if p.TotalSamples() != total {
		t.Fatal("wrong total")
	}
	previousEnd := int64(0)
	for i := int64(0); i < p.Count(); i++ {
		w, err := p.At(i)
		if err != nil {
			t.Fatal(err)
		}
		if w.Index != i || w.Start < 0 || w.End > total || w.End <= w.Start || w.EmitStart != previousEnd || w.EmitEnd <= w.EmitStart || w.EmitStart < w.Start || w.EmitEnd > w.End || w.PadSamples != length-(w.End-w.Start) || w.InputSamples != length {
			t.Fatalf("invalid window %+v previous=%d", w, previousEnd)
		}
		if i > 0 {
			prev, _ := p.At(i - 1)
			if w.Start != prev.End-overlap {
				t.Fatal("analysis overlap differs")
			}
		}
		previousEnd = w.EmitEnd
	}
	if previousEnd != total {
		t.Fatalf("coverage=%d total=%d", previousEnd, total)
	}
}

func TestWindowPlanFullCoverageWithoutHundredChunkCap(t *testing.T) {
	for _, total := range []int64{0, 1, 159, 160, 480000, 480001, 480000 + 464000, 480000 + 464001, 2252 * 16000, 4 * 3600 * 16000} {
		for _, overlap := range []int64{0, 1, 16000, 15999} {
			checkWindowCoverage(t, total, 480000, overlap)
		}
	}
	p, _ := NewWindowPlan(4*3600*16000, 480000, 16000)
	if p.Count() <= 100 {
		t.Fatalf("unexpected cutoff: %d", p.Count())
	}
	final, _ := p.At(p.Count() - 1)
	if final.EmitEnd != 4*3600*16000 {
		t.Fatal("last audio tail lost")
	}
	// Small high-overlap geometries check ownership without allocating audio.
	for _, total := range []int64{160, 161, 319, 320, 481} {
		for _, overlap := range []int64{0, 1, 80, 159} {
			checkWindowCoverage(t, total, 160, overlap)
		}
	}
}

func TestWindowPlanBoundsAndMaxIntArithmetic(t *testing.T) {
	for _, v := range [][3]int64{{-1, 480000, 16000}, {1, 0, 0}, {1, 159, 0}, {1, 480001, 0}, {1, 160, -1}, {1, 160, 160}, {1, 160, 1000}} {
		if _, err := NewWindowPlan(v[0], v[1], v[2]); err == nil {
			t.Fatalf("accepted geometry%v", v)
		}
	}
	max := int64(^uint64(0) >> 1)
	for _, overlap := range []int64{0, 1, 16000, 479999} {
		p, err := NewWindowPlan(max, 480000, overlap)
		if err != nil {
			t.Fatal(err)
		}
		for _, i := range []int64{0, p.Count() / 2, p.Count() - 1} {
			w, err := p.At(i)
			if err != nil || w.Start < 0 || w.EmitEnd > w.End || w.End > max || w.EmitEnd <= w.EmitStart {
				t.Fatalf("overflow %+v %v", w, err)
			}
		}
		w, _ := p.At(p.Count() - 1)
		if w.End != max || w.EmitEnd != max {
			t.Fatal("last bound wrong")
		}
		for _, i := range []int64{-1, p.Count(), max} {
			if _, err := p.At(i); err == nil {
				t.Fatal("invalid index accepted")
			}
		}
	}
	var p WindowPlan
	if _, err := p.At(0); err == nil {
		t.Fatal("zero plan accepted")
	}
	p, _ = NewWindowPlan(0, 480000, 0)
	if p.Count() != 0 {
		t.Fatal("empty plan")
	}
}

type sampleReadFunc func(context.Context, []float32, int64) (int, error)

func (f sampleReadFunc) ReadSamplesAt(ctx context.Context, dst []float32, start int64) (int, error) {
	return f(ctx, dst, start)
}

func TestWindowPlanReadPaddingErrorsAndCancellation(t *testing.T) {
	p, _ := NewWindowPlan(321, 320, 80)
	calls := 0
	reader := sampleReadFunc(func(_ context.Context, dst []float32, start int64) (int, error) {
		calls++
		for i := range dst {
			dst[i] = float32(start + int64(i))
		}
		return len(dst), nil
	})
	scratch := make([]float32, 330)
	for i := range scratch {
		scratch[i] = 999
	}
	w, err := p.ReadWindow(context.Background(), reader, 1, scratch)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 || w.Start != 240 || w.End != 321 || w.PadSamples != 239 {
		t.Fatalf("bad window %+v", w)
	}
	for i, v := range scratch {
		switch {
		case i < 81:
			if v != float32(240+i) {
				t.Fatal("bad sample")
			}
		case i < 320:
			if v != 0 {
				t.Fatal("tail not padded")
			}
		default:
			if v != 999 {
				t.Fatal("scratch beyond window overwritten")
			}
		}
	}
	if _, err := p.ReadWindow(context.Background(), reader, 0, scratch[:319]); err == nil {
		t.Fatal("short scratch accepted")
	}
	if _, err := p.ReadWindow(context.Background(), nil, 0, scratch); err == nil {
		t.Fatal("nil source accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.ReadWindow(ctx, reader, 0, scratch); !errors.Is(err, context.Canceled) {
		t.Fatal("precancel failed")
	}
	if calls != 1 {
		t.Fatal("invalid read invoked source")
	}
	for _, result := range []struct {
		n   int
		err error
	}{{0, nil}, {1, io.EOF}, {319, nil}, {-1, nil}, {321, nil}} {
		bad := sampleReadFunc(func(context.Context, []float32, int64) (int, error) { return result.n, result.err })
		if _, err := p.ReadWindow(context.Background(), bad, 0, scratch); !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Fatalf("short source result %v", err)
		}
	}
	sentinel := errors.New("source failed")
	bad := sampleReadFunc(func(context.Context, []float32, int64) (int, error) { return 0, sentinel })
	if _, err := p.ReadWindow(context.Background(), bad, 0, scratch); !errors.Is(err, sentinel) {
		t.Fatalf("lost source error:%v", err)
	}
	during, stop := context.WithCancel(context.Background())
	defer stop()
	late := sampleReadFunc(func(_ context.Context, dst []float32, _ int64) (int, error) { stop(); return len(dst), nil })
	if _, err := p.ReadWindow(during, late, 0, scratch); !errors.Is(err, context.Canceled) {
		t.Fatal("late cancellation ignored")
	}
}

func TestWindowPlanNoPerWindowAllocation(t *testing.T) {
	p, _ := NewWindowPlan(640, 320, 80)
	reader := sampleReadFunc(func(_ context.Context, dst []float32, _ int64) (int, error) { clear(dst); return len(dst), nil })
	scratch := make([]float32, 320)
	ctx := context.Background()
	var err error
	if alloc := testing.AllocsPerRun(10, func() { _, err = p.ReadWindow(ctx, reader, 1, scratch) }); alloc != 0 {
		t.Fatalf("per-window allocations=%g", alloc)
	}
	if err != nil {
		t.Fatal(err)
	}
}

func TestWindowPlanCanonicalMediaAndCheckedFeatures(t *testing.T) {
	// Actual consumer composition, but no encoder/decoder/model/FFmpeg invocation.
	// The final one-frame tail must be right-padded and featured, never dropped.
	const frames = 321
	data := make([]byte, 44+2*frames)
	copy(data, "RIFF")
	binary.LittleEndian.PutUint32(data[4:], uint32(len(data)-8))
	copy(data[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(data[16:], 16)
	binary.LittleEndian.PutUint16(data[20:], 1)
	binary.LittleEndian.PutUint16(data[22:], 1)
	binary.LittleEndian.PutUint32(data[24:], 16000)
	binary.LittleEndian.PutUint32(data[28:], 32000)
	binary.LittleEndian.PutUint16(data[32:], 2)
	binary.LittleEndian.PutUint16(data[34:], 16)
	copy(data[36:], "data")
	binary.LittleEndian.PutUint32(data[40:], 2*frames)
	binary.LittleEndian.PutUint16(data[44+2*(frames-1):], 16384)
	path := filepath.Join(t.TempDir(), "speech.wav")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	reader, err := media.OpenCanonicalPCM(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	var source SampleReader = reader
	plan, err := NewWindowPlan(int64(reader.Timeline().Samples), 320, 0)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Count() != 2 {
		t.Fatal("missing tail window")
	}
	scratch := make([]float32, 320)
	w, err := plan.ReadWindow(context.Background(), source, 1, scratch)
	if err != nil {
		t.Fatal(err)
	}
	if w.End != 321 || w.EmitStart != 320 || scratch[0] != 0.5 || scratch[1] != 0 {
		t.Fatal("bad final window")
	}
	mel, n, err := MelFlatFromSamplesChecked(scratch, LargeV3Turbo())
	if err != nil || n != 2 || len(mel) != 256 {
		t.Fatalf("checked feature composition failed:%v", err)
	}
}
