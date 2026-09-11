package media

import (
	"context"
	"encoding/binary"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func pcmFixture(t *testing.T, frames int) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "canonical.wav")
	writeCanonicalWAV(t, path, frames)
	return path
}

func TestPCMReaderSignedSamplesAndRandomAccess(t *testing.T) {
	path := pcmFixture(t, 5)
	file, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	info, err := validateCanonicalWAVFile(context.Background(), file, 1<<20, 10000)
	if err != nil {
		t.Fatal(err)
	}
	want := []int16{-32768, -1, 0, 1, 32767}
	raw := make([]byte, 2*len(want))
	for i, v := range want {
		binary.LittleEndian.PutUint16(raw[2*i:], uint16(v))
	}
	if _, err := file.WriteAt(raw, info.dataOffset); err != nil {
		t.Fatal(err)
	}
	file.Close()
	r, err := OpenCanonicalPCM(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if got := r.Timeline(); got.SampleRate != 16000 || got.Samples != 5 {
		t.Fatalf("timeline=%+v", got)
	}
	dst := make([]float32, 8)
	for i := range dst {
		dst[i] = 9
	}
	n, err := r.ReadSamplesAt(context.Background(), dst, 0)
	if n != 5 || !errors.Is(err, io.EOF) {
		t.Fatalf("n=%d err=%v", n, err)
	}
	for i, v := range want {
		if dst[i] != float32(v)/32768 {
			t.Fatalf("sample%d=%v", i, dst[i])
		}
	}
	for _, v := range dst[5:] {
		if v != 9 {
			t.Fatal("tail overwritten")
		}
	}
	n, err = r.ReadSamplesAt(context.Background(), dst[:2], 3)
	if n != 2 || err != nil || dst[0] != float32(want[3])/32768 || dst[1] != float32(want[4])/32768 {
		t.Fatal("positional read failed")
	}
	if n, err = r.ReadSamplesAt(context.Background(), dst, 5); n != 0 || !errors.Is(err, io.EOF) {
		t.Fatal("end read failed")
	}
	if n, err = r.ReadSamplesAt(context.Background(), nil, 5); n != 0 || err != nil {
		t.Fatal("empty read failed")
	}
	for _, pos := range []int64{-1, 6, int64(^uint64(0) >> 1)} {
		if _, err := r.ReadSamplesAt(context.Background(), dst, pos); err == nil {
			t.Fatal("accepted invalid offset")
		}
	}
}

func TestPCMReaderMultiBlockAndNoReadAllocations(t *testing.T) {
	path := pcmFixture(t, 20000)
	r, err := OpenCanonicalPCM(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	dst := make([]float32, 20000)
	n, err := r.ReadSamplesAt(context.Background(), dst, 0)
	if n != len(dst) || err != nil {
		t.Fatalf("read %d %v", n, err)
	}
	// More than two conversion blocks; validate against the fixture's s16 bytes.
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	raw := make([]byte, 2*len(dst))
	if _, err := file.ReadAt(raw, r.info.dataOffset); err != nil {
		t.Fatal(err)
	}
	for i, v := range dst {
		if v != float32(int16(binary.LittleEndian.Uint16(raw[2*i:])))/32768 {
			t.Fatalf("different sample at%d", i)
		}
	}
	ctx := context.Background()
	if allocations := testing.AllocsPerRun(10, func() { n, err = r.ReadSamplesAt(ctx, dst, 0) }); allocations != 0 {
		t.Fatalf("read allocations=%g", allocations)
	}
	if err != nil || n != len(dst) {
		t.Fatal("allocation-check read failed")
	}
}

func TestPCMReaderCancellationAndClose(t *testing.T) {
	path := pcmFixture(t, 16)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := OpenCanonicalPCM(ctx, path); !errors.Is(err, context.Canceled) {
		t.Fatal("precancel open")
	}
	r, err := OpenCanonicalPCM(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	dst := []float32{3, 4}
	if n, err := r.ReadSamplesAt(ctx, dst, 0); n != 0 || !errors.Is(err, context.Canceled) || dst[0] != 3 {
		t.Fatal("precancel read")
	}
	// Simulate an owned read holding the gate; waiting must honour cancellation.
	r.gate <- struct{}{}
	waitCtx, done := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer done()
	n, err := r.ReadSamplesAt(waitCtx, dst, 0)
	<-r.gate
	if n != 0 || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("blocked cancellation=%v", err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := r.ReadSamplesAt(context.Background(), dst, 0); !errors.Is(err, ErrPCMReaderClosed) {
		t.Fatal("read after close")
	}
	if r.Timeline().Samples != 16 {
		t.Fatal("lost metadata after close")
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal("close removed file")
	}
	var zero PCMReader
	if _, err := zero.ReadSamplesAt(context.Background(), dst, 0); !errors.Is(err, ErrPCMReaderClosed) {
		t.Fatal("zero reader")
	}
	if zero.Timeline().Valid() {
		t.Fatal("zero timeline valid")
	}
	var nilReader *PCMReader
	if nilReader.Close() != nil {
		t.Fatal("nil close")
	}
}

func TestPCMReaderConcurrentPositionalReads(t *testing.T) {
	r, err := OpenCanonicalPCM(context.Background(), pcmFixture(t, 200))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func(start int64) {
			defer wg.Done()
			dst := make([]float32, 30)
			for j := 0; j < 4; j++ {
				if n, err := r.ReadSamplesAt(context.Background(), dst, start); n != 30 || err != nil {
					t.Errorf("read %d %v", n, err)
				}
			}
		}(int64(i * 30))
	}
	wg.Wait()
}

func TestPCMReaderTruncationIsNotEOF(t *testing.T) {
	path := pcmFixture(t, 200)
	r, err := OpenCanonicalPCM(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if err := os.Truncate(path, r.info.dataOffset+4); err != nil {
		t.Fatal(err)
	}
	dst := []float32{9, 9, 9, 9}
	n, err := r.ReadSamplesAt(context.Background(), dst, 0)
	if n != 0 || !errors.Is(err, ErrInvalidOutput) || errors.Is(err, io.EOF) {
		t.Fatalf("hidden truncation: %d %v", n, err)
	}
	for _, v := range dst {
		if v != 9 {
			t.Fatal("partial bytes exposed as samples")
		}
	}
}

func TestPCMReaderLongSparseFileBounded(t *testing.T) {
	// Four-hour file with sparse silence: create/read metadata and the last frame,
	// no giant in-memory recording, decoding, model or performance run.
	path := pcmFixture(t, 1)
	frames := maxFramesForDuration(DefaultMaxDuration)
	f, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	info, err := validateCanonicalWAVFile(context.Background(), f, 1<<20, 100)
	if err != nil {
		t.Fatal(err)
	}
	size := info.dataOffset + 2*frames
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(size-8))
	if _, err := f.WriteAt(b[:], 4); err != nil {
		t.Fatal(err)
	}
	binary.LittleEndian.PutUint32(b[:], uint32(2*frames))
	if _, err := f.WriteAt(b[:], info.dataOffset-4); err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(size); err != nil {
		t.Fatal(err)
	}
	f.Close()
	r, err := OpenCanonicalPCM(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if int64(r.Timeline().Samples) != frames || len(r.bytes) != 16*1024 {
		t.Fatal("unbounded reader or wrong timeline")
	}
	dst := []float32{1}
	if n, err := r.ReadSamplesAt(context.Background(), dst, frames-1); n != 1 || err != nil || dst[0] != 0 {
		t.Fatal("last frame lost")
	}
}

func TestCanonicalWAVChunkScanBound(t *testing.T) {
	path := pcmFixture(t, 2)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// Empty chunks cannot force an unbounded metadata scan.
	for i := 0; i < 4096; i++ {
		data = append(data, []byte("JUNK\x00\x00\x00\x00")...)
	}
	binary.LittleEndian.PutUint32(data[4:8], uint32(len(data)-8))
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenCanonicalPCM(context.Background(), path); !errors.Is(err, ErrInvalidOutput) {
		t.Fatalf("chunk bound error=%v", err)
	}
}
