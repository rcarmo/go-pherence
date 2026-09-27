package needle

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"testing"
	"time"
)

// Opt-in isolated-process probe. Run each mode in a fresh binary process and
// record /usr/bin/time's VmHWM separately; this test does not assert RSS gains.
func TestReleasedNeedleResourceProbe(t *testing.T) {
	mode := os.Getenv("GO_PHERENCE_NEEDLE_RESOURCE_MODE")
	if mode == "" {
		t.Skip("set GO_PHERENCE_NEEDLE_RESOURCE_MODE=decoded|hybrid|direct and pin the local archive")
	}
	if mode != "decoded" && mode != "hybrid" && mode != "direct" {
		t.Fatalf("unknown resource mode %q", mode)
	}
	path := os.Getenv("GO_PHERENCE_NEEDLE_PACKED_MODEL")
	if path == "" {
		t.Fatal("missing GO_PHERENCE_NEEDLE_PACKED_MODEL")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != "c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38" {
		t.Fatalf("unexpected archive sha256 %s", got)
	}
	data = nil
	debug.FreeOSMemory()
	loadStart := time.Now()
	var m *Model
	if mode == "direct" {
		m, _, err = LoadArchivePacked(path)
	} else {
		m, _, err = LoadArchive(path)
	}
	if err != nil {
		t.Fatal(err)
	}
	loadMS := time.Since(loadStart).Milliseconds()
	debug.FreeOSMemory()
	loadRSS := probeStatusKB(t, "VmRSS:")
	opts := Options{Packed: mode != "decoded"}
	ids := []int{2, 7}
	var elapsed []int64
	for i := 0; i < 5; i++ {
		decoder, err := m.NewDecoder(DecoderOptions{Capacity: len(ids), Execution: opts})
		if err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		for _, token := range ids {
			if _, err = decoder.Step(context.Background(), token); err != nil {
				t.Fatal(err)
			}
		}
		elapsed = append(elapsed, time.Since(start).Microseconds())
	}
	debug.FreeOSMemory()
	warmRSS := probeStatusKB(t, "VmRSS:")
	hwm := probeStatusKB(t, "VmHWM:")
	t.Logf("mode=%s load_ms=%d logical_decoded_bytes=%d logical_packed_bytes=%d post_load_rss_kib=%d post_warm_rss_kib=%d in_process_hwm_kib=%d two_token_decode_us=%v", mode, loadMS, m.DecodedBytes(), m.PackedBytes(), loadRSS, warmRSS, hwm, elapsed)
	runtime.KeepAlive(m)
}

// This opt-in benchmark times one bounded twelve-token cached session after
// model loading and decoder construction. Run each mode in a separate process.
func BenchmarkReleasedNeedleCachedTwelve(b *testing.B) {
	mode := os.Getenv("GO_PHERENCE_NEEDLE_RESOURCE_MODE")
	path := os.Getenv("GO_PHERENCE_NEEDLE_PACKED_MODEL")
	if mode == "" || path == "" {
		b.Skip("set resource mode and the pinned local archive")
	}
	if mode != "decoded" && mode != "hybrid" && mode != "direct" {
		b.Fatalf("unknown resource mode %q", mode)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		b.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != "c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38" {
		b.Fatalf("unexpected archive sha256 %s", got)
	}
	var m *Model
	if mode == "direct" {
		m, _, err = LoadArchivePacked(path)
	} else {
		m, _, err = LoadArchive(path)
	}
	if err != nil {
		b.Fatal(err)
	}
	ids := []int{2, 7, 4, 9, 3, 6, 5, 8, 7, 3, 5, 2}
	decoder, err := m.NewDecoder(DecoderOptions{Capacity: len(ids), Execution: Options{Packed: mode != "decoded"}})
	if err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(int64(len(ids)))
	b.ResetTimer()
	for b.Loop() {
		decoder.Reset()
		for _, id := range ids {
			if _, err = decoder.Step(ctx, id); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func probeStatusKB(t *testing.T, key string) int64 {
	t.Helper()
	status, err := os.ReadFile("/proc/self/status")
	if err != nil {
		t.Fatal(err)
	}
	for _, line := range strings.Split(string(status), "\n") {
		if strings.HasPrefix(line, key) {
			fields := strings.Fields(strings.TrimPrefix(line, key))
			if len(fields) != 2 || fields[1] != "kB" {
				t.Fatalf("unexpected %s status %q", key, line)
			}
			n, err := strconv.ParseInt(fields[0], 10, 64)
			if err != nil {
				t.Fatal(err)
			}
			return n
		}
	}
	t.Fatalf("missing %s in /proc/self/status", key)
	return 0
}
