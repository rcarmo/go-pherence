package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"reflect"
	"strings"
	"syscall"
	"time"
	"unicode/utf8"

	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	model "github.com/rcarmo/go-pherence/models/omnivoice"
)

type serveOptions struct {
	shared                                                           bool
	modelPath, weightsPath, reference, outputDir, language, instruct string
	frames, firstFrames, steps, workers                              int
	residentBytes, cacheBytes                                        int64
	prepacked, denoise, postprocess                                  bool
}

type phraseEntry struct {
	wave []float32
	used uint64
}

// Process-local LRU. Configuration/reference/model are fixed for its lifetime.
// Byte budget covers waveform payload; at most 256 entries bound map/key cost.
type phraseCache struct {
	entries       map[[32]byte]phraseEntry
	budget, bytes int64
	clock         uint64
}

func (c *phraseCache) get(key [32]byte) ([]float32, bool) {
	e, ok := c.entries[key]
	if !ok {
		return nil, false
	}
	c.clock++
	e.used = c.clock
	c.entries[key] = e
	return e.wave, true
}
func (c *phraseCache) put(key [32]byte, wave []float32) {
	n := int64(len(wave)) * 4
	if n == 0 || n > c.budget {
		return
	}
	if c.entries == nil {
		c.entries = make(map[[32]byte]phraseEntry)
	}
	if old, ok := c.entries[key]; ok {
		c.bytes -= int64(len(old.wave)) * 4
		delete(c.entries, key)
	}
	for c.bytes+n > c.budget || len(c.entries) >= 256 {
		var victim [32]byte
		oldest := ^uint64(0)
		for k, e := range c.entries {
			if e.used < oldest {
				oldest = e.used
				victim = k
			}
		}
		c.bytes -= int64(len(c.entries[victim].wave)) * 4
		delete(c.entries, victim)
	}
	c.clock++
	c.entries[key] = phraseEntry{wave: append([]float32(nil), wave...), used: c.clock}
	c.bytes += n
}
func phraseKey(p loader.PreparedPrompt) [32]byte {
	return sha256.Sum256([]byte(fmt.Sprintf("%d:%s", p.TargetFrames, p.Text)))
}

type serveRequest struct {
	ID     string `json:"id"`
	Text   string `json:"text"`
	Frames int    `json:"frames,omitempty"`
}
type serveEngine struct {
	plan         func(string) ([]loader.PreparedPrompt, error)
	prepareFixed func(string, int) (loader.PreparedPrompt, error)
	maxFrames    int
	generate     func(context.Context, loader.PreparedPrompt, *chunkTimings) ([]float32, error)
	cache        phraseCache
	outputDir    string
	retained     int
}

// Sequential NDJSON protocol, backpressure via synchronous writes. Each emitted
// chunk WAV has a 5ms edge fade and (except the last) a 100ms trailing gap.
// Gain limiting is per chunk; concatenate playback in index order. No final WAV
// is accumulated in memory. Files are retained for the caller, even if a later
// chunk/request fails. This is a local pipe worker, not a network listener.
func (e *serveEngine) loop(ctx context.Context, input io.Reader, output io.Writer) error {
	scanner := bufio.NewScanner(input)
	scanner.Buffer(make([]byte, 4096), 64<<10)
	enc := json.NewEncoder(output)
	sequence := 0
	for scanner.Scan() {
		if err := ctx.Err(); err != nil {
			return err
		}
		sequence++
		e.retained = 0
		var req serveRequest
		dec := json.NewDecoder(bytes.NewReader(scanner.Bytes()))
		dec.DisallowUnknownFields()
		err := dec.Decode(&req)
		if err == nil {
			var extra any
			if tail := dec.Decode(&extra); tail != io.EOF {
				err = fmt.Errorf("one JSON object per line required")
			}
		}
		if err == nil && (len(req.ID) > 128 || !utf8.ValidString(req.Text) || strings.TrimSpace(req.Text) == "") {
			err = fmt.Errorf("id <=128 bytes and nonempty UTF-8 text required")
		}
		if err == nil {
			err = e.request(ctx, enc, req, sequence)
		}
		if err != nil {
			if _, ok := err.(*serveOutputError); ok {
				return err
			}
			if sendErr := enc.Encode(map[string]any{"event": "error", "id": req.ID, "sequence": sequence, "error": err.Error(), "partial_chunks_retained": e.retained > 0, "retained_chunks": e.retained}); sendErr != nil {
				return sendErr
			}
			if ctx.Err() != nil {
				return ctx.Err()
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := scanner.Err(); err != nil {
		if writeErr := enc.Encode(map[string]any{"event": "error", "fatal": true, "error": "input read failed or NDJSON line exceeds 64 KiB", "partial_chunks_retained": false}); writeErr != nil {
			return writeErr
		}
		return err
	}
	return nil
}

type serveOutputError struct{ error }

func sendServe(enc *json.Encoder, v any) error {
	if err := enc.Encode(v); err != nil {
		return &serveOutputError{err}
	}
	return nil
}
func (e *serveEngine) request(ctx context.Context, enc *json.Encoder, req serveRequest, sequence int) error {
	started := time.Now()
	var prompts []loader.PreparedPrompt
	var err error
	if req.Frames != 0 {
		if req.Frames < 1 || req.Frames > e.maxFrames || e.prepareFixed == nil {
			return fmt.Errorf("frames must be 1..configured frames, or omitted for automatic planning")
		}
		var p loader.PreparedPrompt
		p, err = e.prepareFixed(req.Text, req.Frames)
		prompts = []loader.PreparedPrompt{p}
	} else {
		prompts, err = e.plan(req.Text)
	}
	if err != nil {
		return err
	}
	planning := time.Since(started).Seconds()
	if len(prompts) == 0 || len(prompts) > 128 {
		return fmt.Errorf("invalid chunk count")
	}
	totalFrames := 0
	for _, p := range prompts {
		if p.TargetFrames < 1 || p.TargetFrames > 250 {
			return fmt.Errorf("invalid chunk frames")
		}
		totalFrames += p.TargetFrames
	}
	if totalFrames > 15000 {
		return fmt.Errorf("chunk plan exceeds 10 minutes")
	}
	if err := sendServe(enc, map[string]any{"event": "start", "id": req.ID, "sequence": sequence, "chunks": len(prompts), "planning_seconds": planning}); err != nil {
		return err
	}
	hits := 0
	firstSeconds := float64(0)
	audioSeconds := float64(0)
	for i, p := range prompts {
		if err := ctx.Err(); err != nil {
			return err
		}
		at := time.Now()
		var timing chunkTimings
		key := phraseKey(p)
		wave, hit := e.cache.get(key)
		if hit {
			hits++
		} else {
			wave, err = e.generate(ctx, p, &timing)
			if err != nil {
				return fmt.Errorf("chunk %d: %w", i, err)
			}
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		// join copies its inputs: cached/generated raw audio must never be faded twice.
		prepared, err := joinChunkWaves([][]float32{wave}, 120, 0)
		if err != nil {
			return err
		}
		if i < len(prompts)-1 {
			prepared = append(prepared, make([]float32, 2400)...)
		}
		file := filepath.Join(e.outputDir, fmt.Sprintf("%06d-%03d.wav", sequence, i))
		written := time.Now()
		gain, err := loader.WriteSyntheticWAV(file, prepared, 24000)
		if err != nil {
			return err
		}
		writeSeconds := time.Since(written).Seconds()
		e.retained++
		// Cache only successful validated outputs, privately copied within budget.
		if !hit {
			e.cache.put(key, wave)
		}
		elapsed := time.Since(started).Seconds()
		if i == 0 {
			firstSeconds = elapsed
		}
		seconds := float64(len(prepared)) / 24000
		audioSeconds += seconds
		if err := sendServe(enc, map[string]any{"event": "chunk", "synthetic": true, "id": req.ID, "sequence": sequence, "index": i, "text": p.Text, "target_frames": p.TargetFrames, "path": file, "audio_seconds": seconds, "cache_hit": hit, "stages": timing, "write_seconds": writeSeconds, "chunk_seconds": time.Since(at).Seconds(), "elapsed_seconds": elapsed, "gain": gain, "sample_rate": 24000}); err != nil {
			return err
		}
	}
	return sendServe(enc, map[string]any{"event": "done", "id": req.ID, "sequence": sequence, "chunks": len(prompts), "cache_hits": hits, "cache_payload_bytes": e.cache.bytes, "first_chunk_seconds": firstSeconds, "request_seconds": time.Since(started).Seconds(), "audio_seconds": audioSeconds})
}

func runServe(o serveOptions) error {
	started := time.Now()
	if o.modelPath == "" || o.reference == "" || o.outputDir == "" || o.frames < 1 || o.frames > 250 || o.firstFrames < 0 || o.firstFrames > o.frames || o.steps < 1 || o.steps > 128 || o.cacheBytes < 0 || o.cacheBytes > 256<<20 {
		return fmt.Errorf("serve requires model, cached reference, existing output-dir, frames 1..250, first-frames 0..frames, steps 1..128, cache-mib 0..256")
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-ctx.Done():
			_ = os.Stdin.Close()
		case <-done:
		}
	}()
	at := time.Now()
	cfg, err := loader.LoadConfig(o.modelPath)
	if err != nil {
		return err
	}
	tok, err := tokenizer.Load(filepath.Join(o.modelPath, "tokenizer.json"))
	if err != nil {
		return err
	}
	info, err := os.Stat(o.reference)
	if err != nil {
		return err
	}
	if info.Size() > 16<<20 {
		return fmt.Errorf("reference cache exceeds 16 MiB")
	}
	ref, err := loader.LoadCachedReferenceTokens(o.reference)
	if err != nil {
		return err
	}
	referenceSeconds := time.Since(at).Seconds()
	plan := func(text string) ([]loader.PreparedPrompt, error) {
		return loader.PlanChunksWithFirstLimit(cfg, tok, text, ref, loader.PreparePromptOptions{Language: o.language, Instruct: o.instruct, Denoise: o.denoise}, o.frames, o.firstFrames)
	}
	if _, err := plan("Ready."); err != nil {
		return err
	}
	at = time.Now()
	weightsPath := o.weightsPath
	if weightsPath == "" {
		weightsPath = o.modelPath
	}
	weights, err := loader.OpenWeights(weightsPath)
	if err != nil {
		return err
	}
	defer weights.Close()
	if !reflect.DeepEqual(cfg, weights.Config) {
		return fmt.Errorf("weights config differs from model config")
	}
	weightsSeconds := time.Since(at).Seconds()
	at = time.Now()
	cw, err := loader.LoadCodecDecoder(filepath.Join(o.modelPath, "audio_tokenizer"))
	if err != nil {
		return err
	}
	decoder, err := model.NewCodecDecoder(cw)
	if err != nil {
		return err
	}
	codecSeconds := time.Since(at).Seconds()
	at = time.Now()
	// Reserve fixed protocol limits once. Dummy prompt has capacities only; it is
	// never passed to inference. Real requests supply validated prepared prompts.
	capacity := loader.PreparedPrompt{TargetFrames: o.frames}
	capacity.Conditional.Tokens = 512
	capacity.Unconditional.Tokens = o.frames
	runner, err := newChunkRunnerWithResident(ctx, weights, decoder, []loader.PreparedPrompt{capacity}, o.steps, o.residentBytes, o.prepacked, o.workers, o.shared)
	if err != nil {
		return err
	}
	defer runner.cond.Close()
	runnerSeconds := time.Since(at).Seconds()
	dir, err := os.MkdirTemp(o.outputDir, "omnivoice-")
	if err != nil {
		return err
	}
	dir, err = filepath.Abs(dir)
	if err != nil {
		return err
	}
	engine := serveEngine{plan: plan, maxFrames: o.frames, prepareFixed: func(text string, frames int) (loader.PreparedPrompt, error) {
		if utf8.RuneCountInString(text) > 16000 {
			return loader.PreparedPrompt{}, fmt.Errorf("text exceeds 16000 runes")
		}
		return loader.PrepareInferenceInputs(cfg, tok, strings.TrimSpace(text), frames, ref, loader.PreparePromptOptions{Language: o.language, Instruct: o.instruct, Denoise: o.denoise})
	}, generate: func(ctx context.Context, p loader.PreparedPrompt, t *chunkTimings) ([]float32, error) {
		return runner.generateTimed(ctx, p, o.postprocess, t)
	}, cache: phraseCache{budget: o.cacheBytes}, outputDir: dir}
	if err := json.NewEncoder(os.Stdout).Encode(map[string]any{"event": "ready", "protocol": 1, "transport": "ndjson-stdio", "synthetic": true, "output_dir": dir, "startup_seconds": time.Since(started).Seconds(), "reference_tokenizer_seconds": referenceSeconds, "weights_seconds": weightsSeconds, "codec_load_seconds": codecSeconds, "runner_setup_seconds": runnerSeconds, "resident_cache_bytes": runner.cond.ResidentBytes(), "prepacked_bytes": runner.cond.PrepackedBytes(), "cache_budget_bytes": o.cacheBytes, "first_frames": o.firstFrames, "max_frames": o.frames, "steps": o.steps, "gemm_workers": o.workers, "shared_traversal": o.shared, "boundary_fade_ms": 5, "boundary_gap_ms": 100}); err != nil {
		return err
	}
	return engine.loop(ctx, os.Stdin, os.Stdout)
}
