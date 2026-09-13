package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"strings"
	"testing"

	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
)

func TestPhraseCacheBoundedLRU(t *testing.T) {
	c := phraseCache{budget: 16}
	a, b, d := [32]byte{1}, [32]byte{2}, [32]byte{3}
	input := []float32{1, 2}
	c.put(a, input)
	input[0] = 9
	c.put(b, []float32{3, 4})
	got, ok := c.get(a)
	if !ok || got[0] != 1 {
		t.Fatal("ownership")
	}
	c.put(d, []float32{5, 6})
	if _, ok := c.get(b); ok {
		t.Fatal("LRU eviction")
	}
	if c.bytes != 16 {
		t.Fatal(c.bytes)
	}
	c.put(d, []float32{7})
	if c.bytes != 12 {
		t.Fatal("replacement")
	}
	c.put(b, make([]float32, 5))
	if _, ok := c.get(b); ok {
		t.Fatal("oversized")
	}
	disabled := phraseCache{}
	disabled.put(a, input)
	if _, ok := disabled.get(a); ok {
		t.Fatal("disabled")
	}
	for i := 0; i < 300; i++ {
		key := [32]byte{byte(i), byte(i >> 8)}
		c.put(key, []float32{1})
	}
	if c.bytes > c.budget {
		t.Fatal("budget")
	}
}
func testServeEngine(t *testing.T) (*serveEngine, *int) {
	t.Helper()
	calls := 0
	e := &serveEngine{outputDir: t.TempDir(), cache: phraseCache{budget: 1 << 20}}
	e.plan = func(text string) ([]loader.PreparedPrompt, error) {
		return []loader.PreparedPrompt{{Text: "first", TargetFrames: 1}, {Text: "second", TargetFrames: 1}}, nil
	}
	e.generate = func(ctx context.Context, p loader.PreparedPrompt, timing *chunkTimings) ([]float32, error) {
		calls++
		timing.DenoiseSeconds = .1
		w := make([]float32, 960)
		for i := range w {
			w[i] = float32(math.Sin(float64(i)*.1)) * .1
		}
		return w, nil
	}
	return e, &calls
}
func decodeEvents(t *testing.T, b []byte) []map[string]any {
	t.Helper()
	var out []map[string]any
	dec := json.NewDecoder(bytes.NewReader(b))
	for {
		var event map[string]any
		err := dec.Decode(&event)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, event)
	}
	return out
}
func TestServeWarmCacheProgressiveAndValidation(t *testing.T) {
	e, calls := testServeEngine(t)
	var out bytes.Buffer
	original := e.generate
	e.generate = func(ctx context.Context, p loader.PreparedPrompt, timing *chunkTimings) ([]float32, error) {
		if p.Text == "second" && !strings.Contains(out.String(), `"event":"chunk"`) {
			t.Fatal("first chunk not emitted before second inference")
		}
		return original(ctx, p, timing)
	}
	in := "{\"id\":\"a\",\"text\":\"hello\"}\n{\"id\":\"a\",\"text\":\"hello\"}\n{\"text\":\"oops\",\"unknown\":true}\n{} {}\n"
	if err := e.loop(context.Background(), strings.NewReader(in), &out); err != nil {
		t.Fatal(err)
	}
	events := decodeEvents(t, out.Bytes())
	if len(events) != 10 || *calls != 2 {
		t.Fatalf("events %d calls %d", len(events), *calls)
	}
	if events[7]["cache_hits"] != float64(2) {
		t.Fatal(events[7])
	}
	for _, pair := range [][2]int{{1, 5}, {2, 6}} {
		a, err := os.ReadFile(events[pair[0]]["path"].(string))
		if err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(events[pair[1]]["path"].(string))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(a, b) {
			t.Fatal("cached wave changed")
		}
	}
	if events[8]["event"] != "error" || events[9]["event"] != "error" {
		t.Fatal("bad requests accepted")
	}
}

type brokenServeWriter struct{}

func (brokenServeWriter) Write([]byte) (int, error) { return 0, errors.New("broken pipe") }
func TestServeFailureAndCancel(t *testing.T) {
	e, calls := testServeEngine(t)
	if err := e.loop(context.Background(), strings.NewReader("{\"text\":\"hello\"}\n"), brokenServeWriter{}); err == nil {
		t.Fatal("write error ignored")
	}
	if *calls != 0 {
		t.Fatal("computed after output failure")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := e.loop(ctx, strings.NewReader("{\"text\":\"hello\"}\n"), io.Discard); err != context.Canceled {
		t.Fatal(err)
	}
	e.generate = func(context.Context, loader.PreparedPrompt, *chunkTimings) ([]float32, error) {
		return []float32{float32(math.NaN())}, nil
	}
	var out bytes.Buffer
	if err := e.loop(context.Background(), strings.NewReader("{\"text\":\"hello\"}\n"), &out); err != nil {
		t.Fatal(err)
	}
	if e.cache.bytes != 0 || !strings.Contains(out.String(), `"event":"error"`) {
		t.Fatal("invalid waveform cached")
	}
	if err := e.loop(context.Background(), strings.NewReader(strings.Repeat("x", 70<<10)), io.Discard); err == nil {
		t.Fatal("oversized line accepted")
	}
}

func TestServeErrorRetainedFiles(t *testing.T) {
	e, _ := testServeEngine(t)
	var out bytes.Buffer
	if err := e.loop(context.Background(), strings.NewReader("{\"bad\":true}\n"), &out); err != nil {
		t.Fatal(err)
	}
	events := decodeEvents(t, out.Bytes())
	if events[0]["partial_chunks_retained"] != false {
		t.Fatal(events)
	}
	out.Reset()
	original := e.generate
	e.generate = func(ctx context.Context, p loader.PreparedPrompt, timing *chunkTimings) ([]float32, error) {
		if p.Text == "second" {
			return nil, errors.New("second failed")
		}
		return original(ctx, p, timing)
	}
	if err := e.loop(context.Background(), strings.NewReader("{\"text\":\"ok\"}\n"), &out); err != nil {
		t.Fatal(err)
	}
	events = decodeEvents(t, out.Bytes())
	last := events[len(events)-1]
	if last["partial_chunks_retained"] != true || last["retained_chunks"] != float64(1) {
		t.Fatal(events)
	}
	out.Reset()
	if err := e.loop(context.Background(), strings.NewReader(strings.Repeat("x", 70<<10)), &out); err == nil {
		t.Fatal("oversize")
	}
	events = decodeEvents(t, out.Bytes())
	if events[0]["fatal"] != true {
		t.Fatal(events)
	}
}
