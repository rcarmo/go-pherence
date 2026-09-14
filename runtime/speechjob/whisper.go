//go:build linux && amd64

package speechjob

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"github.com/rcarmo/go-pherence/loader/audio/media"
	"github.com/rcarmo/go-pherence/models/whisper"
)

// WhisperStageConfig binds a caller-owned, immutable Go model to raw window
// checkpoints. ModelSHA256 and RuntimeSHA256 are caller attestations to the
// loaded weights and runtime/ISA/backend flags; this constructor cannot recover
// weight provenance from in-memory tensors. Tokenizer/config/options are hashed.
// The model must be loaded without GPU buffers; NVIDIA must be disabled before
// process initialisation and throughout execution. No resident Vulkan is used.
// Empty GenerationJSON uses the checked path's explicit options/model defaults.
type WhisperStageConfig struct {
	ModelSHA256, RuntimeSHA256             string
	Language                               string
	OverlapSamples                         int64
	MaxNewTokens, MaxInitialTimestampIndex int
	SkipDigitalSilence                     bool
	WordTimestamps                         bool
	GenerationJSON                         []byte
	MaxWindowBytes, MaxResultBytes         int64
}

// NewWhisperWindowStage emits "asr-windows" JSONL, NOT final transcript JSON or
// VTT. Each line is a raw Whisper WindowTranscript in canonical PCM seconds.
// Journals acknowledge each window durably before inference advances. Retry
// verifies all acknowledged prefix bytes and starts at the first missing window.
// The source checkpoint must be named "decode". Models/tokenizer stay caller
// owned and immutable; no model loading, workers, services or fallback is hidden.
func NewWhisperWindowStage(model *whisper.Whisper, tokenizer *whisper.Tokenizer, cfg WhisperStageConfig) (Stage, error) {
	return newWhisperWindowStage(model, tokenizer, cfg, nil)
}

// residentStageBinding is private: callers cannot inject an alternate neural
// runtime through the public Vulkan constructor. Tests use this lifetime seam.
type residentStageBinding struct {
	identity string
	validate func() error
	encoder  *whisper.VulkanEncoder
	wrap     func(windowInfer) windowInfer
}

func newWhisperWindowStage(model *whisper.Whisper, tokenizer *whisper.Tokenizer, cfg WhisperStageConfig, resident *residentStageBinding) (Stage, error) {
	if model == nil || tokenizer == nil || tokenizer.VocabSize != model.Config.VocabSize || len(tokenizer.Vocab) != model.Config.VocabSize || !validHash(cfg.ModelSHA256) || !validHash(cfg.RuntimeSHA256) || cfg.MaxWindowBytes < 1 || cfg.MaxWindowBytes > 1<<20 || cfg.MaxResultBytes < cfg.MaxWindowBytes || cfg.MaxResultBytes > 64<<20 || len(cfg.GenerationJSON) > maxConfig {
		return Stage{}, ErrConfiguration
	}
	validate := model.ValidatePCMHostOnly
	if resident != nil {
		validate = resident.validate
	}
	if e := validate(); e != nil {
		return Stage{}, e
	}
	opts := whisper.PCMTranscribeOptions{Language: cfg.Language, OverlapSamples: cfg.OverlapSamples, MaxNewTokens: cfg.MaxNewTokens, MaxInitialTimestampIndex: cfg.MaxInitialTimestampIndex, SkipDigitalSilence: cfg.SkipDigitalSilence, WordTimestamps: cfg.WordTimestamps}
	if resident != nil {
		opts.VulkanEncoder = resident.encoder
	}
	if cfg.WordTimestamps && len(cfg.GenerationJSON) == 0 {
		return Stage{}, ErrConfiguration
	}
	if len(cfg.GenerationJSON) > 0 {
		generation, e := whisper.ParseGenerationConfigChecked(cfg.GenerationJSON, model.Config, tokenizer)
		if e != nil {
			return Stage{}, e
		}
		if cfg.WordTimestamps && !generation.SupportsWordAlignment() {
			return Stage{}, ErrConfiguration
		}
		opts.Generation = generation
	}
	if _, e := whisper.NewWindowPlan(0, int64(model.Config.MaxLength)*160, cfg.OverlapSamples); e != nil || cfg.OverlapSamples > int64(model.Config.MaxLength)*80 || cfg.MaxNewTokens < 0 || cfg.MaxNewTokens > model.Config.MaxDecoderLength-3 || cfg.MaxInitialTimestampIndex < 0 || cfg.MaxInitialTimestampIndex > 1500 || len(cfg.Language) < 2 || len(cfg.Language) > 32 {
		return Stage{}, ErrConfiguration
	}
	for _, c := range cfg.Language {
		if c < 'a' || c > 'z' {
			return Stage{}, ErrConfiguration
		}
	}
	// Checked inference validates the full vocabulary and generation policy before
	// reading samples or running any neural operator. Construction also requires
	// immutable caller ownership; it reads model metadata and tokenizer contents.
	identity, e := json.Marshal(struct {
		Schema          string
		Config          WhisperStageConfig
		Model           whisper.Config
		Tokenizer       *whisper.Tokenizer
		Suppress, Begin []int
	}{"speechjob-go-whisper-windows-word-zero-span-v2", cfg, model.Config, tokenizer, model.Decoder.SuppressTokens, model.Decoder.BeginSuppressTokens})
	if e != nil {
		return Stage{}, e
	}
	version := hash(identity)
	if resident != nil {
		b, e := json.Marshal(struct{ Schema, Host, Resident string }{"speechjob-go-whisper-vulkan-windows-v1", version, resident.identity})
		if e != nil {
			return Stage{}, e
		}
		version = hash(b)
	}
	infer := windowInfer(func(ctx context.Context, source whisper.SampleReader, total, first int64, emit func(whisper.WindowTranscript) error) error {
		if e := validate(); e != nil {
			return e
		}
		return model.TranscribePCMWindowsFrom(ctx, source, total, tokenizer, opts, first, emit)
	})
	if resident != nil {
		infer = resident.wrap(infer)
	}
	return whisperWindowStage(version, int64(model.Config.MaxLength)*160, cfg.OverlapSamples, cfg.MaxWindowBytes, cfg.MaxResultBytes, model.Config.MaxDecoderLength, model.Config.VocabSize, infer), nil
}

type windowInfer func(context.Context, whisper.SampleReader, int64, int64, func(whisper.WindowTranscript) error) error

// windowRecord binds payloads to the full stage dependency key and geometry.
// Its hash is separately recorded in an atomic acknowledgement. A published
// payload without acknowledgement is an orphan: infer again and compare bytes.
type windowRecord struct {
	Schema int                      `json:"schema"`
	Key    string                   `json:"key"`
	Result whisper.WindowTranscript `json:"result"`
}
type windowAck struct {
	Schema int    `json:"schema"`
	Key    string `json:"key"`
	Index  int64  `json:"index"`
	Blob   Blob   `json:"blob"`
}

func windowBase(key string, index int64) string { return fmt.Sprintf("window-%s-%05d", key, index) }

func whisperWindowStage(version string, length, overlap, windowLimit, resultLimit int64, maxTokens, vocab int, infer windowInfer) Stage {
	return Stage{Name: "asr-windows", Version: version, Run: func(ctx context.Context, in *Input, out io.Writer) (err error) {
		var decoded Blob
		for _, cp := range in.job.Checkpoints {
			if cp.Stage == "decode" {
				decoded = cp.Blob
			}
		}
		if decoded.File == "" {
			return fmt.Errorf("missing decode checkpoint")
		}
		// Executor has verified dependencies; verify again before the path-based PCM
		// opener. The private store and its payloads must remain immutable throughout.
		verified, e := in.store.openBlob(ctx, in.job.ID, decoded)
		if e != nil {
			return e
		}
		if e = verified.Close(); e != nil {
			return e
		}
		pcm, e := media.OpenCanonicalPCM(ctx, filepath.Join(in.store.root.Name(), in.job.ID, decoded.File))
		if e != nil {
			return e
		}
		defer func() { err = errors.Join(err, pcm.Close()) }()
		total := int64(pcm.Timeline().Samples)
		plan, e := whisper.NewWindowPlan(total, length, overlap)
		if e != nil {
			return e
		}
		if overlap > length/2 || plan.Count() > 10000 {
			return ErrLimit
		}
		key := checkpointKey(in.job, Stage{Name: "asr-windows", Version: version})
		// Reserve final stream + all new window payloads + bounded acknowledgements.
		// Existing journals/orphans already count; conservative double reservation on
		// retry is intentional. This bounds files, not model memory or CPU resources.
		used, _, e := in.store.usage()
		if e != nil {
			return e
		}
		if resultLimit > in.store.limits.MaxArtifactBytes || 2*resultLimit+plan.Count()*4096+maxManifest > in.store.limits.MaxBytes-used {
			return ErrLimit
		}
		entries, e := os.ReadDir(filepath.Join(in.store.root.Name(), in.job.ID))
		if e != nil {
			return e
		}
		acks := make(map[int64]bool)
		for _, entry := range entries {
			name := entry.Name()
			if !strings.HasPrefix(name, "window-"+key+"-") || !strings.HasSuffix(name, ".ack") {
				continue
			}
			var index int64
			if _, e = fmt.Sscanf(strings.TrimSuffix(strings.TrimPrefix(name, "window-"+key+"-"), ".ack"), "%d", &index); e != nil || index < 0 || index >= plan.Count() || name != windowBase(key, index)+".ack" || entry.IsDir() {
				return ErrCorrupt
			}
			acks[index] = true
		}
		var next, bytes int64
		emitStored := func(data []byte) error {
			if int64(len(data)) > resultLimit-bytes {
				return ErrLimit
			}
			if e := writeContext(ctx, out, data); e != nil {
				return e
			}
			bytes += int64(len(data))
			return nil
		}
		for next < plan.Count() && acks[next] {
			data, e := in.store.readWindow(ctx, in.job.ID, key, next, plan, windowLimit, maxTokens, vocab)
			if e != nil {
				return e
			}
			if e = emitStored(data); e != nil {
				return e
			}
			next++
		}
		if int64(len(acks)) != next {
			return fmt.Errorf("%w: non-prefix window journal", ErrCorrupt)
		}
		publish := func(result whisper.WindowTranscript) error {
			if e := ctx.Err(); e != nil {
				return e
			}
			if result.Window.Index != next {
				return fmt.Errorf("%w: out-of-order ASR window", ErrCorrupt)
			}
			if e := validateWindow(result, plan, maxTokens, vocab); e != nil {
				return e
			}
			data, e := json.Marshal(windowRecord{Schema: 1, Key: key, Result: result})
			if e != nil {
				return e
			}
			data = append(data, '\n')
			if int64(len(data)) > windowLimit || int64(len(data)) > resultLimit-bytes {
				return ErrLimit
			}
			if e = in.store.publishWindow(ctx, in.job.ID, key, next, data, windowLimit); e != nil {
				return e
			}
			next++
			return emitStored(data)
		}
		var callbackErr error
		e = infer(ctx, pcm, total, next, func(result whisper.WindowTranscript) error {
			if callbackErr != nil {
				return callbackErr
			}
			callbackErr = publish(result)
			return callbackErr
		})
		if e != nil || callbackErr != nil {
			return errors.Join(e, callbackErr)
		}
		if next != plan.Count() {
			return fmt.Errorf("%w: incomplete ASR window stream", ErrCorrupt)
		}
		return ctx.Err()
	}}
}
func validateWindow(result whisper.WindowTranscript, plan whisper.WindowPlan, maxTokens, vocab int) error {
	expected, e := plan.At(result.Window.Index)
	if e != nil || result.Window != expected {
		return fmt.Errorf("%w: window geometry", ErrCorrupt)
	}
	if len(result.Language) > 32 {
		return fmt.Errorf("%w: window language", ErrCorrupt)
	}
	for _, c := range result.Language {
		if c < 'a' || c > 'z' {
			return fmt.Errorf("%w: window language", ErrCorrupt)
		}
	}
	if len(result.Segments) > maxTokens {
		return ErrLimit
	}
	last := float64(expected.Start) / 16000
	tokens := 0
	for _, s := range result.Segments {
		if math.IsNaN(s.Start) || math.IsNaN(s.End) || math.IsInf(s.Start, 0) || math.IsInf(s.End, 0) || s.Start < last || s.End <= s.Start || s.End > float64(expected.End)/16000 || len(s.Text) > 65536 || !utf8.ValidString(s.Text) {
			return fmt.Errorf("%w: window segment", ErrCorrupt)
		}
		tokens += len(s.Tokens)
		if tokens > maxTokens {
			return ErrLimit
		}
		for _, t := range s.Tokens {
			if t < 0 || t >= vocab {
				return ErrCorrupt
			}
		}
		last = s.End
	}
	wordTokenEnd, wordEnd := 0, float64(expected.Start)/16000
	for _, word := range result.Words {
		if math.IsNaN(word.Start) || math.IsNaN(word.End) || math.IsInf(word.Start, 0) || math.IsInf(word.End, 0) || word.Start < wordEnd || word.End < word.Start || word.End > float64(expected.End)/16000 || word.TokenStart != wordTokenEnd || word.TokenEnd <= word.TokenStart || word.TokenEnd > tokens || len(word.Word) == 0 || len(word.Word) > 65536 || !utf8.ValidString(word.Word) || strings.TrimSpace(word.Word) == "" {
			return fmt.Errorf("%w: window word", ErrCorrupt)
		}
		wordTokenEnd, wordEnd = word.TokenEnd, word.End
	}
	if len(result.Words) > 0 && wordTokenEnd != tokens {
		return fmt.Errorf("%w: incomplete window word tokens", ErrCorrupt)
	}
	return nil
}
func (s *Store) publishWindow(ctx context.Context, id, key string, index int64, data []byte, limit int64) error {
	base := windowBase(key, index)
	blob, e := s.writeBlob(ctx, id, base, limit, func(w io.Writer) error { return writeContext(ctx, w, data) })
	if e != nil {
		return e
	}
	if e = s.hit("window-payload-published"); e != nil {
		return e
	}
	encoded, e := json.Marshal(windowAck{Schema: 1, Key: key, Index: index, Blob: blob})
	if e != nil {
		return e
	}
	// writeBlob gives acknowledgement bytes the same fsync/no-clobber/dir-sync
	// discipline, including byte-identical orphan conflict detection on retry.
	_, e = s.writeBlob(ctx, id, base+".ack", 4096, func(w io.Writer) error { return writeContext(ctx, w, encoded) })
	if e != nil {
		return errors.Join(ErrPersistence, e)
	}
	return s.hit("window-ack-published")
}
func (s *Store) readWindow(ctx context.Context, id, key string, index int64, plan whisper.WindowPlan, limit int64, maxTokens, vocab int) ([]byte, error) {
	base := windowBase(key, index)
	f, e := s.root.Open(id + "/" + base + ".ack")
	if e != nil {
		return nil, e
	}
	st, e := f.Stat()
	if e != nil {
		f.Close()
		return nil, e
	}
	if !st.Mode().IsRegular() || st.Size() > 4096 {
		f.Close()
		return nil, ErrCorrupt
	}
	raw, e := io.ReadAll(io.LimitReader(f, 4097))
	e = errors.Join(e, f.Close())
	if e != nil {
		return nil, e
	}
	var ack windowAck
	if e = json.Unmarshal(raw, &ack); e != nil {
		return nil, ErrCorrupt
	}
	canonical, _ := json.Marshal(ack)
	if string(canonical) != string(raw) || ack.Schema != 1 || ack.Key != key || ack.Index != index || ack.Blob.File != base || ack.Blob.Bytes > limit {
		return nil, ErrCorrupt
	}
	r, e := s.openBlob(ctx, id, ack.Blob)
	if e != nil {
		return nil, e
	}
	data, e := io.ReadAll(io.LimitReader(r, limit+1))
	e = errors.Join(e, r.Close())
	if e != nil {
		return nil, e
	}
	var record windowRecord
	if e = json.Unmarshal(data, &record); e != nil {
		return nil, ErrCorrupt
	}
	canonical, _ = json.Marshal(record)
	canonical = append(canonical, '\n')
	if string(canonical) != string(data) || record.Schema != 1 || record.Key != key {
		return nil, ErrCorrupt
	}
	if e = validateWindow(record.Result, plan, maxTokens, vocab); e != nil {
		return nil, e
	}
	if record.Result.Window.Index != index {
		return nil, ErrCorrupt
	}
	return data, nil
}
