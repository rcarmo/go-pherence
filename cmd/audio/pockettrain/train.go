//go:build linux

package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"syscall"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/model/pockettts"
)

type preparedCorpus struct {
	Version int                             `json:"version"`
	Source  *preparedProvenance             `json:"source,omitempty"`
	Rows    []pockettts.PreparedTrainingRow `json:"rows"`
}
type preparedProvenance struct {
	ModelSHA256     string `json:"model_sha256"`
	ConfigSHA256    string `json:"config_sha256"`
	TokenizerSHA256 string `json:"tokenizer_sha256"`
	ManifestSHA256  string `json:"manifest_sha256"`
	Policy          string `json:"policy"`
}

func trainingRequested(args []string) bool {
	for i, arg := range args {
		if arg == "-mode=train" || arg == "--mode=train" || (arg == "-mode" || arg == "--mode") && i+1 < len(args) && args[i+1] == "train" {
			return true
		}
	}
	return false
}
func trainingMain(args []string, out io.Writer) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	return trainPrepared(ctx, args, out)
}
func hashText(s string) string { h := sha256.Sum256([]byte(s)); return hex.EncodeToString(h[:]) }
func trainPrepared(ctx context.Context, args []string, out io.Writer) (result error) {
	fs := flag.NewFlagSet("pockettrain train", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	mode := fs.String("mode", "train", "train")
	backend := fs.String("backend", "simd", "simd or explicit ptx-/vulkan- attention, flow, transformer, optimizer, resident, resident-weight or resident-projection or resident-sequence or resident-time or resident-eos or resident-loss or resident-branch or resident-step or resident-pruned or resident-globalnorm or resident-devicescale or resident-sampleweight or resident-lifetime or resident-statecheck or resident-zeroconst or resident-eosfused or resident-audiofused or resident-voicefused or resident-conditioner or resident-backbone or resident-eosbackbone or resident-diagfused or resident-endpointfused or resident-primaryfused or resident-distillfused or resident-forwardfused or resident-diagforward or resident-weightroots or resident-stepweight or resident-stepbackbone or resident-stepresults or resident-distillresults or resident-eosforward or resident-forwardresults or resident-embeddingindex or resident-embeddingcompact or resident-diagadmission or resident-distilladmission hybrid")
	deviceOptin := fs.Bool("allow-experimental-device-training", false, "explicitly permit native hybrid training; no full GPU training")
	devicePin := fs.String("device-runtime-sha256", "", "required caller pin of device/driver/runtime/numerical implementation")
	maxDevice := fs.Int64("max-device-bytes", 256<<20, "logical byte ceiling (resident mode separately caps resident state and each tape; combined peak can reach twice this)")
	config := fs.String("config", "", "released English YAML")
	model := fs.String("model", "", "released safetensors including frozen Mimi")
	modelHash := fs.String("model-sha256", "", "required checkpoint pin")
	corpus := fs.String("prepared-corpus", "", "version1 prepared latent JSON; no recordings decoded")
	output := fs.String("output-dir", "", "existing exclusive-writer checkpoint directory")
	export := fs.String("export", "", "optional NEW output safetensors for inference")
	resume := fs.Bool("resume", false, "resume committed bundle in output-dir")
	experimental := fs.Bool("allow-experimental-sampler", false, "required; sampler numerics/distribution/voice quality unqualified")
	revision := fs.String("source-revision", "", "required full source revision binding this training build")
	steps := fs.Int("steps", 1, "additional completed steps,1..100000")
	every := fs.Int("checkpoint-every", 1, "completed steps between checkpoint generations")
	maxBytes := fs.Int64("max-corpus-bytes", 64<<20, "prepared JSON byte ceiling")
	maxResident := fs.Int64("max-resident-bytes", 6<<30, "estimated resident plus snapshot ceiling")
	maxCheckpoint := fs.Int64("max-checkpoint-bytes", 2<<30, "serialized model ceiling")
	seed := fs.Uint64("seed", 42, "run PCG seed")
	lr := fs.Float64("learning-rate", 2e-4, "constant AdamW LR")
	ema := fs.Float64("ema-decay", .999, "EMA decay")
	maxNorm := fs.Float64("max-gradient-norm", 1, "global clipping ceiling")
	if err := fs.Parse(args); err != nil {
		return err
	}
	scope, err := trainingBackendScope(*backend, *deviceOptin, *devicePin, *maxDevice)
	if err != nil {
		return err
	}
	if !*experimental {
		return fmt.Errorf("training requires -allow-experimental-sampler; numerical/distribution/audio qualification is unfinished")
	}
	if fs.NArg() != 0 || *mode != "train" || *steps <= 0 || *steps > 100000 || *every <= 0 || *output == "" || *config == "" || *model == "" || *corpus == "" || !validDigest(*modelHash) || *maxBytes <= 0 || *maxBytes > 256<<20 || *maxResident <= 0 || *maxCheckpoint <= 0 || *maxCheckpoint > 16<<30 || !validRevision(*revision) {
		return fmt.Errorf("invalid training options/ceilings")
	}
	if math.IsNaN(*lr) || math.IsInf(*lr, 0) || float32(*lr) <= 0 || math.IsInf(float64(float32(*lr)), 0) || math.IsNaN(*ema) || *ema < 0 || *ema >= 1 || float32(*ema) >= 1 || math.IsNaN(*maxNorm) || math.IsInf(*maxNorm, 0) || float32(*maxNorm) <= 0 || math.IsInf(float64(float32(*maxNorm)), 0) {
		return fmt.Errorf("invalid optimiser/EMA/gradient options")
	}
	if *export != "" {
		if _, err := os.Lstat(*export); err == nil || !os.IsNotExist(err) {
			return fmt.Errorf("export destination must be a new path")
		}
	}
	st, err := os.Lstat(*output)
	if err != nil || !st.IsDir() || st.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("output-dir must be an existing nonsymlink directory")
	}
	cfgData, cfgAsset, err := boundedBytes(*config, 1<<20)
	if err != nil {
		return err
	}
	cfg, err := pockettts.ParseConfig(cfgData)
	if err != nil {
		return err
	}
	data, corpusAsset, err := boundedBytes(*corpus, *maxBytes)
	if err != nil {
		return err
	}
	declaration, err := decodePrepared(data)
	if err != nil {
		return err
	}
	rows := declaration.Rows
	if declaration.Source == nil || declaration.Source.ModelSHA256 != *modelHash || declaration.Source.ConfigSHA256 != cfgAsset.SHA256 || !validDigest(declaration.Source.TokenizerSHA256) || !validDigest(declaration.Source.ManifestSHA256) || (declaration.Source.Policy != preparedPolicy && declaration.Source.Policy != preparedWholePolicy) {
		return fmt.Errorf("prepared corpus model/config/tokenizer/preprocessing provenance missing or mismatched")
	}
	// Charge parsed/copy corpus peaks independently of model/tape estimates.
	if corpusAsset.Bytes > *maxResident/4 {
		return fmt.Errorf("prepared corpus peaks exceed resident ceiling")
	}
	availableResident := *maxResident - 4*corpusAsset.Bytes
	mappingBudget := availableResident
	// Admit all row geometry and snapshot estimates before materialising model.
	for i, row := range rows {
		shape := pockettts.TrainingShape{MicroBatchRows: 1, TargetFrames: row.Batch.Frames, VoiceFrames: row.Batch.VoiceFrames, TextTokens: len(row.Batch.TextTokens), GradientAccumulation: 1, FlowBatchMultiplier: 1}
		plan, err := pockettts.PlanTrainingShape(cfg, shape, pockettts.ProductionTrainingShapeLimits(availableResident))
		if err != nil {
			return fmt.Errorf("row%d: %w", i, err)
		}
		// A state snapshot copies params/buffers/moments/EMA. Charge a second
		// parameter-state set in addition to the training planner estimate.
		if plan.ParameterBytes > (availableResident-plan.ResidentUpperBytes)/5 {
			return fmt.Errorf("row%d: training plus checkpoint snapshot exceeds resident ceiling", i)
		}
		mappingBudget = min(mappingBudget, availableResident-plan.ResidentUpperBytes-5*plan.ParameterBytes)
		if scope == "vulkan-resident-sharedworkspace-hybrid-v1" {
			// Coherent native allocations count against process RAM, in addition
			// to the independent owned completed-state mutation guard. Reserve
			// both separately capped resident/tape sets, not only CPU scratch.
			mapping, err := sharedWorkspaceHostMappingBudget(plan, availableResident, *maxDevice)
			if err != nil {
				return fmt.Errorf("row%d: %w", i, err)
			}
			mappingBudget = min(mappingBudget, mapping)
		}
		if err := validatePreparedRow(row, cfg.Mimi.InnerDim, cfg.FlowLM.LookupTable.NBins); err != nil {
			return fmt.Errorf("row%d: %w", i, err)
		}
	}
	// Exclusive process writer; stale lock is retained on process death and
	// requires deliberate operator disposition, never automatic lock stealing.
	lockPath := filepath.Join(*output, "runner.lock")
	lock, err := os.OpenFile(lockPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY|syscall.O_NOFOLLOW, 0600)
	if err != nil {
		return fmt.Errorf("exclusive training writer: %w", err)
	}
	defer func() { lock.Close(); os.Remove(lockPath) }()
	if !*resume {
		if _, err := os.Lstat(filepath.Join(*output, "current.json")); err == nil || !os.IsNotExist(err) {
			return fmt.Errorf("existing bundle requires -resume or a new output directory")
		}
	}
	modelFile, size, err := boundedFile(*model, 16<<30)
	if err != nil {
		return err
	}
	defer modelFile.Close()
	h := sha256.New()
	n, readErr := io.Copy(h, io.LimitReader(modelFile, size+1))
	if n != size {
		return fmt.Errorf("model changed size during hashing")
	}
	if readErr != nil {
		return readErr
	}
	if hex.EncodeToString(h.Sum(nil)) != *modelHash {
		return fmt.Errorf("model pin mismatch")
	}
	if size > mappingBudget {
		return fmt.Errorf("checkpoint mapping plus corpus/training/snapshot estimate exceeds resident admission")
	}
	// Linux fd path keeps the same admitted inode across hash and mmap;
	// caller must still keep source contents immutable during this run.
	source, err := safetensors.Open(fmt.Sprintf("/proc/self/fd/%d", modelFile.Fd()))
	if err != nil {
		return err
	}
	defer source.Close()
	lm, err := pockettts.LoadFlowLMTrainingCPU(source, cfg)
	if err != nil {
		return err
	}
	flow, err := pockettts.LoadFlowHeadTrainingCPU(source, cfg)
	if err != nil {
		return err
	}
	adam := pockettts.DefaultAdamWConfig()
	adam.LearningRate = float32(*lr)
	trainer, err := pockettts.NewFullTrainer(lm, flow, pockettts.NewDefaultLSDWeightMLP(), adam, float32(*ema))
	if err != nil {
		return err
	}
	owners, cleanup, err := openTrainingBackend(ctx, scope, *maxDevice)
	if err != nil {
		if cleanup != nil {
			return &trainingResourcesRetainedError{err: err, owner: cleanup}
		}
		return err
	} // Never retry partial failed-cleanup construction.
	defer func() {
		if closeErr := cleanup(); closeErr != nil {
			result = &trainingResourcesRetainedError{err: errors.Join(result, closeErr), owner: owners}
		}
	}()
	lm.AttentionBackwardBackend = owners.attention
	lm.TransformerBackend = owners.transformer
	trainer.OptimizerBackend = owners.optimizer
	trainer.Weighting.TrainingBackend = owners.weight
	lm.ProjectionBackend = owners.projection
	lm.SequenceBackend = owners.sequence
	lm.EOSBackend = owners.eos
	flow.LossBackend = owners.loss
	flow.BranchBackend = owners.branch
	flow.StepReductionBackend = owners.step
	flow.TrainingBackend = owners.flow
	if scope != "simd" {
		if err := json.NewEncoder(out).Encode(map[string]string{"notice": trainingBackendNotice(scope)}); err != nil {
			return err
		}
	}
	identity := pockettts.TrainingRunIdentity{SourceRevision: *revision, GoVersion: runtime.Version(), GOOS: runtime.GOOS, GOARCH: runtime.GOARCH, SIMDMode: fmt.Sprintf("%s-vec%t-dot%t-bf16%t", simd.RuntimeCapabilities().Arch, simd.RuntimeCapabilities().HasVec, simd.RuntimeCapabilities().HasDot, simd.RuntimeCapabilities().HasBF16), Backend: scope, CorpusSHA256: corpusAsset.SHA256, InitialModelSHA256: *modelHash, TokenizerSHA256: declaration.Source.TokenizerSHA256, MimiSHA256: *modelHash, ConfigSHA256: cfgAsset.SHA256, OptimizerSHA256: hashText(fmt.Sprintf("%+v ema=%g clip=%g", adam, *ema, *maxNorm)), OptionsSHA256: hashText(pockettts.TrainingRandomAlgorithm + ";always1;dropout0;row1;fbm1;minimal;normalized;logit.4/1;caller-normalized-latents;seed=" + fmt.Sprint(*seed))}
	identity.OptionsSHA256 = trainingBackendOptionsHash(identity.OptionsSHA256, scope, *devicePin)
	runner, err := pockettts.NewTrainingRunner(trainer, identity, rows, [6]uint64{*seed, *seed + 1, *seed + 2, *seed + 3, *seed + 4, *seed + 5}, pockettts.TrainingRandomLimits{Samples: pockettts.TrainingSampleLimits{MaxFrames: 375, MaxNoiseElements: 375 * 32}, MaxRawDraws: 2*375*32 + 6*375}, float32(*maxNorm), true)
	if err != nil {
		return err
	}
	limits := pockettts.TrainingBundleLimits{MaxModelBytes: *maxCheckpoint, MaxSidecarBytes: 2 << 20}
	if *resume {
		if err := runner.Restore(*output, limits); err != nil {
			return err
		}
	}
	if err := runPreparedUpdates(ctx, runner, *output, *steps, *every, limits, out, trainingGeneration); err != nil {
		return err
	}
	if *export != "" {
		return exportNewCheckpoint(*export, cfg, trainer, source, *ema > 0)
	}
	return nil
}

// The CLI consumes only the declared preparation policy: one zero EOS row,
// raw voice latents and a finite normalised target prefix. The lower-level
// runner separately permits explicit caller-provided mask policies.
func validatePreparedRow(row pockettts.PreparedTrainingRow, channels, vocabulary int) error {
	b := row.Batch
	if channels <= 0 || vocabulary <= 0 || b.Frames < 2 || b.Frames > pockettts.ProductionTrainingTargetFrames || b.VoiceFrames <= 0 || b.VoiceFrames > pockettts.ProductionTrainingVoiceFrames || row.ValidFrames != b.Frames-1 || len(b.TextTokens) == 0 || len(b.TextTokens) > 512 {
		return fmt.Errorf("invalid prepared geometry or one-frame EOS policy")
	}
	// Production channels are small, but use checked arithmetic here since
	// this helper is also directly exercised without config parsing.
	target, ok := checked.MulInt(b.Frames, channels)
	voice, voiceOK := checked.MulInt(b.VoiceFrames, channels)
	if !ok || !voiceOK || len(b.NormalizedLatents) != target || len(b.VoiceLatents) != voice {
		return fmt.Errorf("invalid prepared latent shape")
	}
	for _, values := range [][]float32{b.NormalizedLatents, b.VoiceLatents} {
		for _, x := range values {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				return fmt.Errorf("nonfinite prepared latent")
			}
		}
	}
	for _, x := range b.NormalizedLatents[target-channels:] {
		if x != 0 {
			return fmt.Errorf("prepared EOS row must be zero")
		}
	}
	for _, token := range b.TextTokens {
		if uint64(token) >= uint64(vocabulary) {
			return fmt.Errorf("prepared token outside vocabulary")
		}
	}
	return nil
}

// The loop only saves after Step has returned a successful committed update.
// A failed Step may have poisoned live parameters and must never be saved.
// Publication failures are returned once, without retry or rollback.
type preparedUpdateRunner interface {
	Step(context.Context) (pockettts.TrainingStepMetrics, error)
	Progress() (uint64, uint64, int)
	Checkpoint(string, string, pockettts.TrainingBundleLimits) (pockettts.TrainingCheckpointBinding, error)
}

func trainingGeneration(step uint64) (string, error) {
	var id [8]byte
	if _, err := rand.Read(id[:]); err != nil {
		return "", err
	}
	return fmt.Sprintf("step-%d-%x", step, id), nil
}

func runPreparedUpdates(ctx context.Context, runner preparedUpdateRunner, directory string, steps, every int, limits pockettts.TrainingBundleLimits, out io.Writer, generation func(uint64) (string, error)) error {
	if ctx == nil || runner == nil || out == nil || generation == nil || steps <= 0 || every <= 0 {
		return fmt.Errorf("invalid prepared update loop")
	}
	lastSaved, _, _ := runner.Progress()
	saveDirty := func() error {
		step, _, _ := runner.Progress()
		if step == lastSaved {
			return nil
		}
		name, err := generation(step)
		if err != nil {
			return err
		}
		if _, err := runner.Checkpoint(directory, name, limits); err != nil {
			return err
		}
		lastSaved = step
		return nil
	}
	enc := json.NewEncoder(out)
	for i := 0; i < steps; i++ {
		if err := ctx.Err(); err != nil {
			return errors.Join(err, saveDirty())
		}
		metrics, err := runner.Step(ctx)
		if err != nil {
			// TrainingRunner returns context cancellation only before an update.
			// A cancellation racing the loop's check can therefore save earlier
			// dirty boundaries; other Step failures may be poisoned and cannot.
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return errors.Join(err, saveDirty())
			}
			return err
		}
		step, epoch, cursor := runner.Progress()
		// Commit a due/final bundle before reporting progress. Between bundle
		// intervals the JSON explicitly identifies the last saved step.
		if step%uint64(every) == 0 || i == steps-1 {
			if err := saveDirty(); err != nil {
				return err
			}
		}
		if err := enc.Encode(map[string]any{"step": step, "checkpoint_step": lastSaved, "epoch": epoch, "cursor": cursor, "loss": metrics.Loss, "flow": metrics.FlowLoss, "eos": metrics.EOS, "experimental": true}); err != nil {
			return errors.Join(err, saveDirty())
		}
	}
	return ctx.Err()
}

func validRevision(s string) bool {
	if len(s) != 40 {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}
func parsePrepared(data []byte) ([]pockettts.PreparedTrainingRow, error) {
	c, err := decodePrepared(data)
	return c.Rows, err
}
func decodePrepared(data []byte) (preparedCorpus, error) {
	var c preparedCorpus
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return c, err
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		return c, fmt.Errorf("prepared corpus trailing content")
	}
	if c.Version != 1 || len(c.Rows) == 0 || len(c.Rows) > 100000 {
		return c, fmt.Errorf("invalid prepared corpus version/rows")
	}
	return c, nil
}

// Export through an owned scratch directory, then exclusively link the finished
// file into place. The existing exporter may rename-overwrite its scratch path;
// it must never replace a raced caller destination.
func exportNewCheckpoint(path string, cfg pockettts.Config, trainer *pockettts.FullTrainer, source *safetensors.File, ema bool) error {
	parent := filepath.Dir(path)
	dir, err := os.MkdirTemp(parent, ".pocket-export-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	temp := filepath.Join(dir, "model.safetensors")
	if err := pockettts.ExportPocketSafetensors(temp, cfg, trainer, source, ema); err != nil {
		return err
	}
	return publishNewExport(temp, path)
}
func publishNewExport(temp, path string) error {
	if err := os.Link(temp, path); err != nil {
		return err
	}
	d, err := os.Open(filepath.Dir(path))
	if err != nil {
		return &pockettts.PocketExportPublishedError{Path: path, Err: err}
	}
	syncErr := d.Sync()
	closeErr := d.Close()
	if syncErr != nil || closeErr != nil {
		return &pockettts.PocketExportPublishedError{Path: path, Err: errors.Join(syncErr, closeErr)}
	}
	return nil
}
