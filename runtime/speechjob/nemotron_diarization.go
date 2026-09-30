//go:build linux && amd64

package speechjob

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"path/filepath"
	"sort"

	"github.com/rcarmo/go-pherence/loader/audio/media"
	nem "github.com/rcarmo/go-pherence/model/nemotrondiarization"
)

const NemotronDiarizationRevision = "a435e9867d79e789e90053f9b6d6834053af564a"
const nemotronSpeakerPolicy = "nemotron-full-turns-maximum-positive-overlap-per-word-v1"

type NemotronDiarizationConfig struct {
	AllowExperimental                         bool
	ModelSHA256, RuntimeSHA256, ModelRevision string
	MaxResultBytes                            int64
}

type NemotronDiarizationDocument struct {
	Schema       int                `json:"schema"`
	Provider     string             `json:"provider"`
	Experimental bool               `json:"experimental"`
	SampleRate   int                `json:"sample_rate"`
	TotalSamples int64              `json:"total_samples"`
	StageKey     string             `json:"stage_key"`
	SourceTiming media.SourceTiming `json:"source_timing"`
	Turns        []nem.Segment      `json:"turns"`
}

type nemotronPCMStream interface {
	AppendPCMContext(context.Context, []float32) ([]float32, error)
	FinishContext(context.Context) ([]float32, error)
}

// NewNemotronDiarizationStage uses CPU SIMD only and shares immutable weights.
// Every attempt starts fresh stream/cache state. No Community-1 checkpoint is
// accepted or imitated. Plain transcript/VTT checkpoints survive any failure.
func NewNemotronDiarizationStage(model *nem.PCMStreamingRequest, cfg NemotronDiarizationConfig) (Stage, error) {
	if model == nil {
		return Stage{}, ErrConfiguration
	}
	if _, err := model.NewCPUStream(); err != nil {
		return Stage{}, ErrConfiguration
	}
	return newNemotronDiarizationStage(cfg, func() (nemotronPCMStream, error) { return model.NewCPUStream() })
}

func newNemotronDiarizationStage(cfg NemotronDiarizationConfig, factory func() (nemotronPCMStream, error)) (Stage, error) {
	if !cfg.AllowExperimental || !validHash(cfg.ModelSHA256) || !validHash(cfg.RuntimeSHA256) || cfg.ModelRevision != NemotronDiarizationRevision || cfg.MaxResultBytes < 1 || cfg.MaxResultBytes > 16<<20 || factory == nil {
		return Stage{}, ErrConfiguration
	}
	identity, _ := json.Marshal(struct {
		Schema string
		Config NemotronDiarizationConfig
	}{"speechjob-nemotron-diarization-cpu-stream-v1", cfg})
	version := hash(identity)
	return Stage{Name: "diarization", Version: version, Run: func(ctx context.Context, in *Input, out io.Writer) (err error) {
		if err := ctx.Err(); err != nil {
			return err
		}
		var decoded Blob
		for _, cp := range in.job.Checkpoints {
			if cp.Stage == "decode" {
				decoded = cp.Blob
			}
		}
		if decoded.File == "" {
			return ErrConfiguration
		}
		r, err := in.store.openBlob(ctx, in.job.ID, decoded)
		if err != nil {
			return err
		}
		if err = r.Close(); err != nil {
			return err
		}
		pcm, err := media.OpenCanonicalPCM(ctx, filepath.Join(in.store.root.Name(), in.job.ID, decoded.File))
		if err != nil {
			return err
		}
		defer func() { err = errors.Join(err, pcm.Close()) }()
		total := int64(pcm.Timeline().Samples)
		if total < 160 || total > 4*3600*16000 {
			return ErrLimit
		}
		used, _, err := in.store.usage()
		if err != nil {
			return err
		}
		if cfg.MaxResultBytes > in.store.limits.MaxArtifactBytes || cfg.MaxResultBytes+maxManifest > in.store.limits.MaxBytes-used {
			return ErrLimit
		}
		stream, err := factory()
		if err != nil {
			return err
		}
		if stream == nil {
			return ErrConfiguration
		}
		// Match the native stream's committed 10-ms frame count. Later
		// uncentred windows omit frames extending beyond EOF.
		expectedFrames := total / 160
		if total >= 16640 {
			expectedFrames = (total-264)/160 + 1
		}
		var emittedFrames int64
		var segments nem.SegmentStream
		turns := make([]nem.Segment, 0)
		consume := func(logits []float32) error {
			if len(logits) == 0 {
				return nil
			}
			if len(logits)%8 != 0 || int64(len(logits)/8) > expectedFrames-emittedFrames {
				return ErrCorrupt
			}
			part, err := segments.Append(logits)
			if err != nil {
				return err
			}
			if len(turns)+len(part) > 100000 {
				return ErrLimit
			}
			turns = append(turns, part...)
			emittedFrames += int64(len(logits) / 8)
			return nil
		}
		scratch := make([]float32, 80000)
		ReportDiarizationProgress(ctx, 0, total, "samples")
		for offset := int64(0); offset < total; {
			if err := ctx.Err(); err != nil {
				return err
			}
			count := int(min(int64(len(scratch)), total-offset))
			n, readErr := pcm.ReadSamplesAt(ctx, scratch[:count], offset)
			if readErr != nil && !errors.Is(readErr, io.EOF) {
				return readErr
			}
			if n != count {
				return io.ErrUnexpectedEOF
			}
			logits, err := stream.AppendPCMContext(ctx, scratch[:count])
			if err != nil {
				return err
			}
			if err = consume(logits); err != nil {
				return err
			}
			offset += int64(count)
			ReportDiarizationProgress(ctx, offset, total, "samples")
		}
		logits, err := stream.FinishContext(ctx)
		if err != nil {
			return err
		}
		if err = consume(logits); err != nil {
			return err
		}
		if emittedFrames != expectedFrames {
			return ErrCorrupt
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		last, err := segments.Finish()
		if err != nil {
			return err
		}
		if len(turns)+len(last) > 100000 {
			return ErrLimit
		}
		turns = append(turns, last...)
		sort.Slice(turns, func(i, j int) bool {
			if turns[i].Start != turns[j].Start {
				return turns[i].Start < turns[j].Start
			}
			return turns[i].Speaker < turns[j].Speaker
		})
		key := checkpointKey(in.job, Stage{Name: "diarization", Version: version})
		d := NemotronDiarizationDocument{Schema: 1, Provider: "nemotron", Experimental: true, SampleRate: 16000, TotalSamples: total, StageKey: key, SourceTiming: pcm.SourceTiming(), Turns: turns}
		if err = validateNemotronDiarizationDocument(ctx, d); err != nil {
			return err
		}
		data, err := json.Marshal(d)
		if err != nil {
			return err
		}
		data = append(data, '\n')
		if int64(len(data)) > cfg.MaxResultBytes {
			return ErrLimit
		}
		return writeContext(ctx, out, data)
	}}, nil
}

func validateNemotronDiarizationDocument(ctx context.Context, d NemotronDiarizationDocument) error {
	if d.Schema != 1 || d.Provider != "nemotron" || !d.Experimental || d.SampleRate != 16000 || d.TotalSamples < 160 || d.TotalSamples > 4*3600*16000 || !validHash(d.StageKey) || d.Turns == nil || len(d.Turns) > 100000 {
		return ErrCorrupt
	}
	if _, err := media.MarshalSourceTimingWAVChunk(d.SourceTiming); err != nil {
		return ErrCorrupt
	}
	lastStart, lastSpeaker := -1.0, -1
	var ends [8]float64
	for _, turn := range d.Turns {
		if err := ctx.Err(); err != nil {
			return err
		}
		if math.IsNaN(turn.Start) || math.IsNaN(turn.End) || math.IsInf(turn.Start, 0) || math.IsInf(turn.End, 0) || turn.Start < 0 || turn.End <= turn.Start || turn.End > float64(d.TotalSamples)/16000 || turn.Speaker < 0 || turn.Speaker >= 8 {
			return ErrCorrupt
		}
		if turn.Start < lastStart || turn.Start == lastStart && turn.Speaker < lastSpeaker || turn.Start < ends[turn.Speaker] {
			return ErrCorrupt
		}
		if math.Abs(turn.Start*100-math.Round(turn.Start*100)) > 1e-7 || math.Abs(turn.End*100-math.Round(turn.End*100)) > 1e-7 {
			return ErrCorrupt
		}
		lastStart, lastSpeaker = turn.Start, turn.Speaker
		ends[turn.Speaker] = turn.End
	}
	return ctx.Err()
}

func ReadNemotronDiarizationJSON(ctx context.Context, r io.Reader) (NemotronDiarizationDocument, error) {
	var d NemotronDiarizationDocument
	if r == nil {
		return d, ErrCorrupt
	}
	var b bytes.Buffer
	if err := writeBoundedRead(ctx, &b, r, 16<<20); err != nil {
		return d, err
	}
	if json.Unmarshal(b.Bytes(), &d) != nil {
		return d, ErrCorrupt
	}
	canonical, err := json.Marshal(d)
	if err != nil || !bytes.Equal(append(canonical, '\n'), b.Bytes()) {
		return d, ErrCorrupt
	}
	return d, validateNemotronDiarizationDocument(ctx, d)
}

func writeBoundedRead(ctx context.Context, out *bytes.Buffer, r io.Reader, limit int) error {
	buf := make([]byte, 32<<10)
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		n, err := r.Read(buf)
		if n > 0 {
			if out.Len()+n > limit {
				return ErrLimit
			}
			out.Write(buf[:n])
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrNoProgress
		}
	}
}
