//go:build linux && amd64

package speechjob

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"path/filepath"

	"github.com/rcarmo/go-pherence/loader/audio/media"
	c1 "github.com/rcarmo/go-pherence/models/speaker/community1"
)

// Community1StageConfig is an explicit experimental host-CPU contract. Hashes
// attest caller-loaded immutable segmentation/embedding/PLDA weights, lowered
// filters and runtime flags/ISA. The adapter cannot prove in-memory provenance.
// No opt-in is inferred from constructing experimental models. Config/modes and
// all attestations enter the stage identity. No defaults or fallback are chosen.
type Community1StageConfig struct {
	AllowExperimental                                                             bool
	SegmentationSHA256, EmbeddingSHA256, PLDASHA256, FiltersSHA256, RuntimeSHA256 string
	PCM                                                                           c1.DiarizationPCMConfig
	SegmentationModes                                                             c1.SegmentationModes
	EmbeddingMode                                                                 c1.WeSpeakerBlockMode
	MaxResultBytes                                                                int64
}

// NewCommunity1Stage creates a whole-result "diarization" stage. It preserves
// previously completed ASR/transcript checkpoints on failure; diarization itself
// restarts from the beginning until the complete result is durably published.
// Clustering depends on all windows; it is never restarted on an isolated suffix.
// Callers own immutable models/PCM and exclude competing model/backend use. This
// does not load weights, launch a neural subprocess, relax a gate or start service.
// The existing experimental 128-window limit is enforced before inference.
func NewCommunity1Stage(model *c1.ExperimentalDiarization, cfg Community1StageConfig) (Stage, error) {
	if model == nil {
		return Stage{}, ErrConfiguration
	}
	if e := validateCommunityConfig(cfg); e != nil {
		return Stage{}, e
	}
	return community1Stage(cfg, func(ctx context.Context, reader c1.DiarizationPCMReader, total int64) (*c1.DiarizationPCMResult, error) {
		return model.RunPCM(ctx, reader, total, cfg.PCM, cfg.SegmentationModes, cfg.EmbeddingMode)
	}), nil
}
func validateCommunityConfig(cfg Community1StageConfig) error {
	if !cfg.AllowExperimental || cfg.MaxResultBytes < 1 || cfg.MaxResultBytes > 16<<20 {
		return ErrConfiguration
	}
	for _, v := range []string{cfg.SegmentationSHA256, cfg.EmbeddingSHA256, cfg.PLDASHA256, cfg.FiltersSHA256, cfg.RuntimeSHA256} {
		if !validHash(v) {
			return ErrConfiguration
		}
	}
	c := cfg.PCM
	m := cfg.SegmentationModes
	if c.WindowSamples < 400 || c.WindowSamples > 160000 || c.StepSamples < 1 || c.StepSamples > c.WindowSamples || c.MinimumEmbeddingSamples < 1 || c.MinimumEmbeddingSamples > c.WindowSamples || c.MinSpeakers < 1 || c.MaxSpeakers < c.MinSpeakers || c.MaxSpeakers > 64 || c.NumSpeakers < 0 || c.NumSpeakers > 64 || c.AHCThreshold < 0 || c.Fa <= 0 || c.Fb <= 0 || c.MinDurationOff < 0 || c.MinDurationOff > 30 {
		return ErrConfiguration
	}
	for _, v := range []float64{c.AHCThreshold, c.Fa, c.Fb, c.MinDurationOff} {
		if !finite(v) {
			return ErrConfiguration
		}
	}
	if c.TiePolicy != c1.RejectAmbiguousTies && c.TiePolicy != c1.LowestIndexTies {
		return ErrConfiguration
	}
	if (m.SincNet != c1.SincNetScalarFMA && m.SincNet != c1.SincNetSIMDFMA) || (m.LSTM != c1.LSTMScalar && m.LSTM != c1.LSTMSIMD) || (m.Head != c1.HeadScalar && m.Head != c1.HeadSIMD) {
		return ErrConfiguration
	}
	if cfg.EmbeddingMode != c1.WeSpeakerBlockScalar && cfg.EmbeddingMode != c1.WeSpeakerBlockSIMD && cfg.EmbeddingMode != c1.WeSpeakerBlockGEMM {
		return ErrConfiguration
	}
	return nil
}
func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// DiarizationDocument stores raw frame-centre turns and diagnostic policy, with
// no clipping, integer-sample rounding, speaker relabelling or text alignment.
// TimelineClasses may include padded count columns; it is distinct from Clusters.
// Even ExclusiveTurns may overlap when MinDurationOff fills gaps. Experimental
// is always true. ConstraintSatisfied=false is retained on silence/single-row
// paths; clustered count failures return ErrSpeakerCountFallbackRequired before
// a result exists. NumSpeakers overrides min/max, matching the model contract.
// Geometry validation is not neural qualification.
type DiarizationDocument struct {
	Schema              int                     `json:"schema"`
	Experimental        bool                    `json:"experimental"`
	SampleRate          int                     `json:"sample_rate"`
	TotalSamples        int64                   `json:"total_samples"`
	StageKey            string                  `json:"stage_key"`
	Policy              c1.DiarizationPCMConfig `json:"policy"`
	Windows             []c1.DiarizationWindow  `json:"windows"`
	SegmentationGrid    c1.SincNetGrid          `json:"segmentation_grid"`
	LocalSpeakers       int                     `json:"local_speakers"`
	EmbeddingDimension  int                     `json:"embedding_dimension"`
	Timeline            DiarizationGrid         `json:"timeline"`
	Path                string                  `json:"path"`
	TrainingRows        int                     `json:"training_rows"`
	Clusters            int                     `json:"clusters"`
	ConstraintSatisfied bool                    `json:"constraint_satisfied"`
	AmbiguousFrames     []int                   `json:"ambiguous_frames"`
	FullTurns           []c1.SpeakerTurn        `json:"full_turns"`
	ExclusiveTurns      []c1.SpeakerTurn        `json:"exclusive_turns"`
}
type DiarizationGrid struct {
	Frames        int     `json:"frames"`
	Classes       int     `json:"classes"`
	Start         float64 `json:"start"`
	FrameDuration float64 `json:"frame_duration"`
	FrameStep     float64 `json:"frame_step"`
}
type communityInfer func(context.Context, c1.DiarizationPCMReader, int64) (*c1.DiarizationPCMResult, error)

func community1Stage(cfg Community1StageConfig, infer communityInfer) Stage {
	identity, _ := json.Marshal(struct {
		Schema string
		Config Community1StageConfig
	}{"speechjob-community1-rawturns-v1", cfg})
	version := hash(identity)
	return Stage{Name: "diarization", Version: version, Run: func(ctx context.Context, in *Input, out io.Writer) (err error) {
		if e := ctx.Err(); e != nil {
			return e
		}
		var decoded Blob
		for _, cp := range in.job.Checkpoints {
			if cp.Stage == "decode" {
				decoded = cp.Blob
			}
		}
		if decoded.File == "" {
			return fmt.Errorf("missing decode checkpoint")
		}
		// Store.Run enforces unique stage names and verifies the dependency chain.
		// Path re-opening requires the same immutable private-root/payload contract
		// as the Whisper adapter; no hostile same-UID/rename protection is claimed.
		r, e := in.store.openBlob(ctx, in.job.ID, decoded)
		if e != nil {
			return e
		}
		if e = r.Close(); e != nil {
			return e
		}
		pcm, e := media.OpenCanonicalPCM(ctx, filepath.Join(in.store.root.Name(), in.job.ID, decoded.File))
		if e != nil {
			return e
		}
		defer func() { err = errors.Join(err, pcm.Close()) }()
		total := int64(pcm.Timeline().Samples)
		if _, e = c1.PlanDiarizationWindows(total, cfg.PCM.WindowSamples, cfg.PCM.StepSamples); e != nil {
			return e
		}
		used, _, e := in.store.usage()
		if e != nil {
			return e
		}
		if cfg.MaxResultBytes > in.store.limits.MaxArtifactBytes || cfg.MaxResultBytes+maxManifest > in.store.limits.MaxBytes-used {
			return ErrLimit
		}
		result, e := infer(ctx, pcm, total)
		if e != nil {
			return e
		}
		if e = ctx.Err(); e != nil {
			return e
		}
		key := checkpointKey(in.job, Stage{Name: "diarization", Version: version})
		document, e := diarizationDocument(ctx, result, cfg.PCM, total, key)
		if e != nil {
			return e
		}
		b, e := json.Marshal(document)
		if e != nil {
			return e
		}
		b = append(b, '\n')
		if int64(len(b)) > cfg.MaxResultBytes {
			return ErrLimit
		}
		if e = in.store.hit("diarization-output-ready"); e != nil {
			return e
		}
		return writeContext(ctx, out, b)
	}}
}
func diarizationDocument(ctx context.Context, r *c1.DiarizationPCMResult, policy c1.DiarizationPCMConfig, total int64, key string) (DiarizationDocument, error) {
	var d DiarizationDocument
	if r == nil || r.Postprocess == nil || r.Postprocess.Timeline == nil {
		return d, ErrCorrupt
	}
	p, t := r.Postprocess, r.Postprocess.Timeline
	d = DiarizationDocument{Schema: 1, Experimental: true, SampleRate: 16000, TotalSamples: total, StageKey: key, Policy: policy, Windows: r.Windows, SegmentationGrid: r.Grid, LocalSpeakers: r.LocalSpeakers, EmbeddingDimension: r.EmbeddingDimension, Timeline: DiarizationGrid{t.Frames, t.Classes, t.Start, t.FrameDuration, t.FrameStep}, Path: p.Path, TrainingRows: p.TrainingRows, Clusters: p.Clusters, ConstraintSatisfied: p.ConstraintSatisfied, AmbiguousFrames: append([]int{}, t.AmbiguousFrames...), FullTurns: append([]c1.SpeakerTurn{}, p.FullTurns...), ExclusiveTurns: append([]c1.SpeakerTurn{}, p.ExclusiveTurns...)}
	if e := validateDiarizationDocument(ctx, d); e != nil {
		return DiarizationDocument{}, e
	}
	return d, nil
}
func validateDiarizationDocument(ctx context.Context, d DiarizationDocument) error {
	if e := ctx.Err(); e != nil {
		return e
	}
	if d.Windows == nil || d.AmbiguousFrames == nil || d.FullTurns == nil || d.ExclusiveTurns == nil {
		return ErrCorrupt
	}
	if d.Schema != 1 || !d.Experimental || d.SampleRate != 16000 || !validHash(d.StageKey) || d.LocalSpeakers < 1 || d.LocalSpeakers > 8 || d.EmbeddingDimension < 1 || d.EmbeddingDimension > 512 || d.TrainingRows < 0 || d.TrainingRows > 512 || d.Clusters < 0 || d.Clusters > 64 {
		return ErrCorrupt
	}
	// Use the same policy validator with inert valid attestations; it runs no model.
	check := Community1StageConfig{AllowExperimental: true, SegmentationSHA256: d.StageKey, EmbeddingSHA256: d.StageKey, PLDASHA256: d.StageKey, FiltersSHA256: d.StageKey, RuntimeSHA256: d.StageKey, PCM: d.Policy, SegmentationModes: c1.SegmentationModes{SincNet: c1.SincNetScalarFMA}, MaxResultBytes: 1}
	if e := validateCommunityConfig(check); e != nil {
		return ErrCorrupt
	}
	windows, e := c1.PlanDiarizationWindows(d.TotalSamples, d.Policy.WindowSamples, d.Policy.StepSamples)
	if e != nil || len(windows) != len(d.Windows) {
		return ErrCorrupt
	}
	for i, w := range windows {
		if w != d.Windows[i] {
			return ErrCorrupt
		}
	}
	g, t := d.SegmentationGrid, d.Timeline
	if g.Frames < 2 || g.Frames > 4096 || g.Step < 1 || g.Step > d.Policy.WindowSamples || g.ReceptiveField < 1 || g.ReceptiveField > 16000 || g.FirstCenter < 0 || g.FirstCenter >= d.Policy.WindowSamples || g.FirstCenter+(g.Frames-1)*g.Step >= d.Policy.WindowSamples {
		return ErrCorrupt
	}
	if t.Start != 0 || t.FrameStep != float64(g.Step)/16000 || t.FrameDuration != float64(g.ReceptiveField)/16000 || t.FrameStep > t.FrameDuration || t.Frames < 2 || t.Frames > 1000000 || t.Classes < 0 || t.Classes > 64 {
		return ErrCorrupt
	}
	// Preserve source float64 evaluation order for the reconstruction extent.
	end := float64(d.Policy.WindowSamples)/16000 + float64(len(windows)-1)*(float64(d.Policy.StepSamples)/16000)
	frames := math.RoundToEven(((end+.5*t.FrameDuration)-.5*t.FrameDuration)/t.FrameStep) + 1
	if frames != float64(t.Frames) || len(d.AmbiguousFrames) > t.Frames {
		return ErrCorrupt
	}
	minSpeakers, maxSpeakers := d.Policy.MinSpeakers, d.Policy.MaxSpeakers
	if d.Policy.NumSpeakers > 0 {
		minSpeakers, maxSpeakers = d.Policy.NumSpeakers, d.Policy.NumSpeakers
	}
	switch d.Path {
	case "silence":
		if d.TrainingRows != 0 || d.Clusters != 0 || t.Classes != 0 || d.ConstraintSatisfied || len(d.FullTurns) != 0 || len(d.ExclusiveTurns) != 0 || len(d.AmbiguousFrames) != 0 {
			return ErrCorrupt
		}
	case "single-training-row":
		if d.TrainingRows != 1 || d.Clusters != 1 {
			return ErrCorrupt
		}
	case "clustered":
		if d.TrainingRows < 2 || d.Clusters < 1 || d.Clusters > d.TrainingRows || !d.ConstraintSatisfied {
			return ErrCorrupt
		}
	default:
		return ErrCorrupt
	}
	if d.Path != "silence" && d.ConstraintSatisfied != (d.Clusters >= minSpeakers && d.Clusters <= maxSpeakers) {
		return ErrCorrupt
	}
	if d.TrainingRows > len(windows)*d.LocalSpeakers || d.Policy.TiePolicy == c1.RejectAmbiguousTies && len(d.AmbiguousFrames) > 0 {
		return ErrCorrupt
	}
	previous := -1
	for _, i := range d.AmbiguousFrames {
		if i <= previous || i >= t.Frames {
			return ErrCorrupt
		}
		previous = i
	}
	// Raw turns use the POSTPROCESS reconstruction grid centres, deliberately
	// distinct from SincNet's FirstCenter (see BinaryActivityToTurns). They may
	// extend beyond real PCM into padded reconstruction. Check a tiny arithmetic
	// guard around grid centres only; never clamp or change numerical model gates.
	minTime := .5 * t.FrameDuration
	maxTime := float64(t.Frames-1)*t.FrameStep + .5*t.FrameDuration
	for _, turns := range [][]c1.SpeakerTurn{d.FullTurns, d.ExclusiveTurns} {
		if len(turns) > 100000 {
			return ErrLimit
		}
		var prior c1.SpeakerTurn
		var ends [64]float64
		for i, turn := range turns {
			if i%256 == 0 {
				if e := ctx.Err(); e != nil {
					return e
				}
			}
			if !finite(turn.Start) || !finite(turn.End) || turn.End-turn.Start <= 1e-6 || turn.Start < minTime-1e-9 || turn.End > maxTime+1e-9 || turn.Speaker < 0 || turn.Speaker >= t.Classes || turn.Start < ends[turn.Speaker] {
				return ErrCorrupt
			}
			if i > 0 && (turn.Start < prior.Start || turn.Start == prior.Start && (turn.End < prior.End || turn.End == prior.End && turn.Speaker <= prior.Speaker)) {
				return ErrCorrupt
			}
			ends[turn.Speaker] = turn.End
			prior = turn
		}
	}
	return ctx.Err()
}

// ReadDiarizationJSON accepts only the canonical bytes emitted by this stage.
// This checkpoint reader is intentionally stricter than a general JSON importer:
// whitespace changes, missing/duplicate/unknown/case-alias keys are rejected by
// exact canonical re-encoding. Size is bounded before parsing. Input stays owned
// by the caller; cancellation between blocking reads is cooperative.
func ReadDiarizationJSON(ctx context.Context, r io.Reader) (DiarizationDocument, error) {
	var d DiarizationDocument
	if r == nil {
		return d, ErrCorrupt
	}
	var b []byte
	buf := make([]byte, 32<<10)
	for {
		if e := ctx.Err(); e != nil {
			return d, e
		}
		n, e := r.Read(buf)
		if n > 0 {
			if len(b)+n > 16<<20 {
				return d, ErrLimit
			}
			b = append(b, buf[:n]...)
		}
		if e == io.EOF {
			break
		}
		if e != nil {
			return d, e
		}
		if n == 0 {
			return d, io.ErrNoProgress
		}
	}
	if e := json.Unmarshal(b, &d); e != nil {
		return DiarizationDocument{}, ErrCorrupt
	}
	canonical, e := json.Marshal(d)
	if e != nil {
		return DiarizationDocument{}, ErrCorrupt
	}
	canonical = append(canonical, '\n')
	if string(canonical) != string(b) {
		return DiarizationDocument{}, ErrCorrupt
	}
	if e = validateDiarizationDocument(ctx, d); e != nil {
		return DiarizationDocument{}, e
	}
	return d, nil
}
