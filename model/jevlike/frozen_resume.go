package jevlike

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
)

const (
	frozenResumeStateVersion     = 1
	frozenResumeShuffleAlgorithm = "seed_plus_epoch_v1"
)

// FrozenRunIdentity binds resumable state to one exact dataset/cache/code tuple.
type FrozenRunIdentity struct {
	CacheID          string `json:"cache_id"`
	TrainSHA256      string `json:"train_sha256"`
	ValidationSHA256 string `json:"validation_sha256"`
	CodeRevision     string `json:"code_revision"`
}

type frozenResumeState struct {
	Version           int               `json:"version"`
	ShuffleAlgorithm  string            `json:"shuffle_algorithm"`
	Reference         string            `json:"reference"`
	Config            Config            `json:"config"`
	TrainConfig       TrainConfig       `json:"train_config"`
	Identity          FrozenRunIdentity `json:"identity"`
	Step              int               `json:"step"`
	Epoch             int               `json:"epoch"`
	BatchOffset       int               `json:"batch_offset"`
	PartialEpochLoss  float64           `json:"partial_epoch_loss"`
	PartialEpochCount int               `json:"partial_epoch_count"`
	CurrentState      []NamedParameter  `json:"current_state"`
	AdamState         []frozenAdamState `json:"adam_state"`
	History           []EpochMetrics    `json:"history"`
	HasBest           bool              `json:"has_best"`
	BestValidationNLL float64           `json:"best_validation_nll,omitempty"`
	BestState         []NamedParameter  `json:"best_state,omitempty"`
}

type frozenAdamState struct {
	Name  string    `json:"name"`
	Shape []int     `json:"shape"`
	M     []float32 `json:"m"`
	V     []float32 `json:"v"`
}

// ChoiceExamplesSHA256 returns a stable SHA-256 over one exact example slice.
func ChoiceExamplesSHA256(examples []ChoiceExample) (string, error) {
	h := sha256.New()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(len(examples)))
	if _, err := h.Write(buf[:]); err != nil {
		return "", err
	}
	for i, example := range examples {
		item, err := ValidateChoiceExample(example)
		if err != nil {
			return "", fmt.Errorf("example %d: %w", i, err)
		}
		if err := frozenResumeHashString(h, item.Context); err != nil {
			return "", err
		}
		binary.LittleEndian.PutUint64(buf[:], uint64(len(item.Options)))
		if _, err := h.Write(buf[:]); err != nil {
			return "", err
		}
		for _, option := range item.Options {
			if err := frozenResumeHashString(h, option); err != nil {
				return "", err
			}
		}
		binary.LittleEndian.PutUint64(buf[:], uint64(item.Label))
		if _, err := h.Write(buf[:]); err != nil {
			return "", err
		}
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func frozenResumeHashString(w io.Writer, s string) error {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(len(s)))
	if _, err := w.Write(buf[:]); err != nil {
		return err
	}
	_, err := io.WriteString(w, s)
	return err
}

// TrainFrozenResumable fits a native frozen scorer, persisting exact optimizer
// state after each batch and evaluation so training can resume bit-exactly.
func TrainFrozenResumable(model *FrozenScorer, train, validation []ChoiceExample, cfg TrainConfig, identity FrozenRunIdentity, statePath string, maxSteps int) (TrainResult, bool, error) {
	cfg, err := frozenResumeResolvedConfig(cfg)
	if err != nil {
		return TrainResult{}, false, err
	}
	if identity.CacheID == "" || identity.CodeRevision == "" || model == nil || model.Reference == "" {
		return TrainResult{}, false, fmt.Errorf("exact cache/code/encoder identity required")
	}
	if statePath == "" {
		return TrainResult{}, false, fmt.Errorf("jevlike resumable state path is empty")
	}
	if maxSteps < 0 {
		return TrainResult{}, false, fmt.Errorf("jevlike max steps must be non-negative")
	}
	if err := validateFrozenScorer(model, true); err != nil {
		return TrainResult{}, false, err
	}
	if len(train) == 0 {
		return TrainResult{}, false, fmt.Errorf("jevlike training set is empty")
	}
	if len(validation) == 0 {
		return TrainResult{}, false, fmt.Errorf("jevlike validation set is empty")
	}
	trainHash, err := ChoiceExamplesSHA256(train)
	if err != nil {
		return TrainResult{}, false, err
	}
	validationHash, err := ChoiceExamplesSHA256(validation)
	if err != nil {
		return TrainResult{}, false, err
	}
	if identity.TrainSHA256 == "" || identity.TrainSHA256 != trainHash {
		return TrainResult{}, false, fmt.Errorf("jevlike training hash mismatch")
	}
	if identity.ValidationSHA256 == "" || identity.ValidationSHA256 != validationHash {
		return TrainResult{}, false, fmt.Errorf("jevlike validation hash mismatch")
	}

	if err := os.MkdirAll(filepath.Dir(statePath), 0o755); err != nil {
		return TrainResult{}, false, err
	}
	lock := statePath + ".lock"
	if err := os.Mkdir(lock, 0o700); err != nil {
		return TrainResult{}, false, fmt.Errorf("trainer lock: %w", err)
	}
	defer os.Remove(lock)
	var state frozenResumeState
	if loaded, err := frozenResumeLoadState(statePath); err == nil {
		state = loaded
		if err := frozenResumeValidateLoadedState(state, model, cfg, identity, len(train)); err != nil {
			return TrainResult{}, false, err
		}
		if err := model.Head.LoadNamedParameters(defaultHeadPrefix, frozenResumeNamedParametersToMap(state.CurrentState)); err != nil {
			return TrainResult{}, false, err
		}
	} else if !os.IsNotExist(err) {
		return TrainResult{}, false, err
	} else {
		if needsFrozenInitialization(model) {
			if err := InitializeFrozenScorer(model, cfg.Seed); err != nil {
				return TrainResult{}, false, err
			}
		}
		state = frozenResumeState{
			Version:          frozenResumeStateVersion,
			ShuffleAlgorithm: frozenResumeShuffleAlgorithm,
			Reference:        model.Reference,
			Config:           model.Config,
			TrainConfig:      cfg,
			Identity:         identity,
			Epoch:            1,
			CurrentState:     model.Head.NamedParameters(defaultHeadPrefix),
			AdamState:        frozenResumeAdamMapToList(make(map[string]adamState), model.Head.NamedParameters(defaultHeadPrefix)),
			BestState:        nil,
		}
	}

	params := model.Head.NamedParameterMap(defaultHeadPrefix)
	states := frozenResumeAdamListToMap(state.AdamState)
	result, finished, err := frozenResumeTrainLoop(model, train, validation, cfg, statePath, maxSteps, state, params, states)
	if err != nil {
		return TrainResult{}, false, err
	}
	return result, finished, nil
}

func frozenResumeTrainLoop(model *FrozenScorer, train, validation []ChoiceExample, cfg TrainConfig, statePath string, maxSteps int, state frozenResumeState, params map[string][]float32, states map[string]adamState) (TrainResult, bool, error) {
	stepsTaken := 0
	for {
		if state.Epoch > cfg.Epochs {
			result := frozenResumeStateToResult(state)
			if result.BestState != nil {
				if err := model.Head.LoadNamedParameters(defaultHeadPrefix, result.BestState); err != nil {
					return TrainResult{}, false, err
				}
			}
			return result, true, nil
		}
		if state.BatchOffset == len(train) {
			if state.PartialEpochCount != len(train) {
				return TrainResult{}, false, fmt.Errorf("jevlike resumable state partial count mismatch")
			}
			validationLoss, err := meanCrossEntropyLossFrozen(model, validation, cfg.BatchSize)
			if err != nil {
				return TrainResult{}, false, err
			}
			state.History = append(state.History, EpochMetrics{
				Epoch:         state.Epoch,
				TrainNLL:      state.PartialEpochLoss / float64(state.PartialEpochCount),
				ValidationNLL: validationLoss,
			})
			if !state.HasBest || validationLoss < state.BestValidationNLL {
				state.HasBest = true
				state.BestValidationNLL = validationLoss
				state.BestState = model.Head.NamedParameters(defaultHeadPrefix)
			}
			state.Epoch++
			state.BatchOffset = 0
			state.PartialEpochLoss = 0
			state.PartialEpochCount = 0
			state.CurrentState = model.Head.NamedParameters(defaultHeadPrefix)
			state.AdamState = frozenResumeAdamMapToList(states, state.CurrentState)
			if err := frozenResumeSaveState(statePath, state); err != nil {
				return TrainResult{}, false, err
			}
			continue
		}
		if maxSteps > 0 && stepsTaken >= maxSteps {
			return frozenResumeStateToResult(state), false, nil
		}

		indices := frozenResumeEpochIndices(cfg.Seed, state.Epoch, len(train))
		start := state.BatchOffset
		end := min(start+cfg.BatchSize, len(indices))
		batchExamples := gatherExamples(train, indices[start:end])
		batch, err := model.encodeBatch(batchExamples)
		if err != nil {
			return TrainResult{}, false, err
		}
		logits, err := model.forwardEncodedBatch(batch, false)
		if err != nil {
			return TrainResult{}, false, err
		}
		loss, dLogits, err := crossEntropyLossAndGradient(logits, batch.OptionMask, batch.Labels)
		if err != nil {
			return TrainResult{}, false, err
		}
		grads, _, _, err := model.Head.Backward(batch.Context, batch.ContextMask, batch.Options, batch.OptionMask, dLogits)
		if err != nil {
			return TrainResult{}, false, err
		}
		clipGradientMapByGlobalNorm(grads, cfg.MaxGradNorm)
		state.Step++
		stepsTaken++
		if err := adamwStep(params, grads, states, state.Step, cfg.LearningRate); err != nil {
			return TrainResult{}, false, err
		}
		if err := model.Head.LoadNamedParameters(defaultHeadPrefix, params); err != nil {
			return TrainResult{}, false, err
		}
		state.BatchOffset = end
		state.PartialEpochLoss += loss * float64(len(batch.Labels))
		state.PartialEpochCount += len(batch.Labels)
		state.CurrentState = model.Head.NamedParameters(defaultHeadPrefix)
		state.AdamState = frozenResumeAdamMapToList(states, state.CurrentState)
		if err := frozenResumeSaveState(statePath, state); err != nil {
			return TrainResult{}, false, err
		}
		if maxSteps > 0 && stepsTaken >= maxSteps {
			return frozenResumeStateToResult(state), false, nil
		}
	}
}

func frozenResumeResolvedConfig(cfg TrainConfig) (TrainConfig, error) {
	out, err := cfg.resolved()
	if err != nil {
		return TrainConfig{}, err
	}
	if math.IsNaN(float64(out.LearningRate)) || math.IsInf(float64(out.LearningRate), 0) || out.LearningRate <= 0 {
		return TrainConfig{}, fmt.Errorf("jevlike learning rate must be positive and finite")
	}
	if math.IsNaN(float64(out.MaxGradNorm)) || math.IsInf(float64(out.MaxGradNorm), 0) || out.MaxGradNorm <= 0 {
		return TrainConfig{}, fmt.Errorf("jevlike max grad norm must be positive and finite")
	}
	return out, nil
}

func frozenResumeLoadState(path string) (frozenResumeState, error) {
	f, err := os.Open(path)
	if err != nil {
		return frozenResumeState{}, err
	}
	defer f.Close()
	var state frozenResumeState
	d := json.NewDecoder(f)
	d.DisallowUnknownFields()
	if err := d.Decode(&state); err != nil {
		return frozenResumeState{}, err
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return frozenResumeState{}, fmt.Errorf("resumable state contains trailing data")
	}
	return state, nil
}

func frozenResumeSaveState(path string, state frozenResumeState) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".jevlike-resume-*.tmp")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	e := json.NewEncoder(f)
	if err := e.Encode(state); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}

func frozenResumeValidateLoadedState(state frozenResumeState, model *FrozenScorer, cfg TrainConfig, identity FrozenRunIdentity, trainLen int) error {
	if state.Version != frozenResumeStateVersion {
		return fmt.Errorf("unsupported jevlike resumable state version %d", state.Version)
	}
	if state.ShuffleAlgorithm != frozenResumeShuffleAlgorithm {
		return fmt.Errorf("unsupported jevlike resumable shuffle algorithm %q", state.ShuffleAlgorithm)
	}
	if state.Reference != model.Reference {
		return fmt.Errorf("jevlike resumable reference mismatch")
	}
	if state.Config != model.Config {
		return fmt.Errorf("jevlike resumable config mismatch")
	}
	stateCfg, err := frozenResumeResolvedConfig(state.TrainConfig)
	if err != nil {
		return err
	}
	if stateCfg != state.TrainConfig || state.TrainConfig != cfg {
		return fmt.Errorf("jevlike resumable training config mismatch")
	}
	if state.Identity != identity {
		return fmt.Errorf("jevlike resumable identity mismatch")
	}
	expected, err := frozenResumeExpectedParameters(state.Config)
	if err != nil {
		return err
	}
	if err := frozenResumeValidateNamedParameters(state.CurrentState, expected, "current_state"); err != nil {
		return err
	}
	if err := frozenResumeValidateAdamState(state.AdamState, expected); err != nil {
		return err
	}
	if trainLen <= 0 {
		return fmt.Errorf("jevlike training set is empty")
	}
	if state.Step < 0 {
		return fmt.Errorf("jevlike resumable step must be non-negative")
	}
	if state.Epoch < 1 || state.Epoch > cfg.Epochs+1 {
		return fmt.Errorf("jevlike resumable epoch %d out of range", state.Epoch)
	}
	if state.BatchOffset < 0 || state.BatchOffset > trainLen {
		return fmt.Errorf("jevlike resumable batch offset %d out of range", state.BatchOffset)
	}
	if state.PartialEpochCount < 0 || state.PartialEpochCount > trainLen {
		return fmt.Errorf("jevlike resumable partial epoch count %d out of range", state.PartialEpochCount)
	}
	if state.PartialEpochCount != state.BatchOffset {
		return fmt.Errorf("jevlike resumable partial epoch count mismatch")
	}
	if math.IsNaN(state.PartialEpochLoss) || math.IsInf(state.PartialEpochLoss, 0) || state.PartialEpochLoss < 0 {
		return fmt.Errorf("jevlike resumable partial epoch loss must be finite and non-negative")
	}
	if state.PartialEpochCount == 0 && state.PartialEpochLoss != 0 {
		return fmt.Errorf("jevlike resumable partial epoch loss/count mismatch")
	}
	completedBatches, err := frozenResumeCompletedBatches(trainLen, cfg.BatchSize, state.BatchOffset)
	if err != nil {
		return err
	}
	fullEpochBatches := (trainLen + cfg.BatchSize - 1) / cfg.BatchSize
	if state.Step != len(state.History)*fullEpochBatches+completedBatches {
		return fmt.Errorf("jevlike resumable step arithmetic mismatch")
	}
	if state.Epoch == cfg.Epochs+1 {
		if state.BatchOffset != 0 || state.PartialEpochCount != 0 || state.PartialEpochLoss != 0 {
			return fmt.Errorf("jevlike resumable completed run has pending epoch state")
		}
		if len(state.History) != cfg.Epochs {
			return fmt.Errorf("jevlike resumable history length mismatch")
		}
	} else if len(state.History) != state.Epoch-1 {
		return fmt.Errorf("jevlike resumable history length mismatch")
	}
	minValidation := 0.0
	for i, metrics := range state.History {
		if metrics.Epoch != i+1 {
			return fmt.Errorf("jevlike resumable history epoch mismatch")
		}
		if math.IsNaN(metrics.TrainNLL) || math.IsInf(metrics.TrainNLL, 0) || metrics.TrainNLL < 0 {
			return fmt.Errorf("jevlike resumable train NLL must be finite and non-negative")
		}
		if math.IsNaN(metrics.ValidationNLL) || math.IsInf(metrics.ValidationNLL, 0) || metrics.ValidationNLL < 0 {
			return fmt.Errorf("jevlike resumable validation NLL must be finite and non-negative")
		}
		if i == 0 || metrics.ValidationNLL < minValidation {
			minValidation = metrics.ValidationNLL
		}
	}
	if len(state.History) == 0 {
		if state.HasBest || len(state.BestState) != 0 || state.BestValidationNLL != 0 {
			return fmt.Errorf("jevlike resumable best state mismatch")
		}
		return nil
	}
	if !state.HasBest {
		return fmt.Errorf("jevlike resumable best state missing")
	}
	if math.IsNaN(state.BestValidationNLL) || math.IsInf(state.BestValidationNLL, 0) || state.BestValidationNLL < 0 {
		return fmt.Errorf("jevlike resumable best validation NLL must be finite and non-negative")
	}
	if state.BestValidationNLL != minValidation {
		return fmt.Errorf("jevlike resumable best validation mismatch")
	}
	if err := frozenResumeValidateNamedParameters(state.BestState, expected, "best_state"); err != nil {
		return err
	}
	return nil
}

func frozenResumeExpectedParameters(cfg Config) ([]NamedParameter, error) {
	head, err := NewAttentionHead(cfg.Width, cfg.Rank)
	if err != nil {
		return nil, err
	}
	return head.NamedParameters(defaultHeadPrefix), nil
}

func frozenResumeValidateNamedParameters(got, expected []NamedParameter, label string) error {
	if len(got) != len(expected) {
		return fmt.Errorf("jevlike resumable %s tensor count=%d, want %d", label, len(got), len(expected))
	}
	for i := range expected {
		if got[i].Name != expected[i].Name {
			return fmt.Errorf("jevlike resumable %s tensor %d name=%q, want %q", label, i, got[i].Name, expected[i].Name)
		}
		if len(got[i].Shape) != len(expected[i].Shape) {
			return fmt.Errorf("jevlike resumable %s tensor %q rank mismatch", label, got[i].Name)
		}
		for j := range expected[i].Shape {
			if got[i].Shape[j] != expected[i].Shape[j] {
				return fmt.Errorf("jevlike resumable %s tensor %q shape mismatch", label, got[i].Name)
			}
		}
		if len(got[i].Values) != len(expected[i].Values) {
			return fmt.Errorf("jevlike resumable %s tensor %q length mismatch", label, got[i].Name)
		}
		for _, value := range got[i].Values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fmt.Errorf("jevlike resumable %s tensor %q contains non-finite value", label, got[i].Name)
			}
		}
	}
	return nil
}

func frozenResumeValidateAdamState(got []frozenAdamState, expected []NamedParameter) error {
	if len(got) != len(expected) {
		return fmt.Errorf("jevlike resumable adam tensor count=%d, want %d", len(got), len(expected))
	}
	for i := range expected {
		if got[i].Name != expected[i].Name {
			return fmt.Errorf("jevlike resumable adam tensor %d name=%q, want %q", i, got[i].Name, expected[i].Name)
		}
		if len(got[i].Shape) != len(expected[i].Shape) {
			return fmt.Errorf("jevlike resumable adam tensor %q rank mismatch", got[i].Name)
		}
		for j := range expected[i].Shape {
			if got[i].Shape[j] != expected[i].Shape[j] {
				return fmt.Errorf("jevlike resumable adam tensor %q shape mismatch", got[i].Name)
			}
		}
		if len(got[i].M) != len(expected[i].Values) || len(got[i].V) != len(expected[i].Values) {
			return fmt.Errorf("jevlike resumable adam tensor %q length mismatch", got[i].Name)
		}
		for _, value := range got[i].M {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fmt.Errorf("jevlike resumable adam tensor %q contains non-finite M", got[i].Name)
			}
		}
		for _, value := range got[i].V {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || value < 0 {
				return fmt.Errorf("jevlike resumable adam tensor %q contains non-finite V", got[i].Name)
			}
		}
	}
	return nil
}

func frozenResumeCompletedBatches(trainLen, batchSize, batchOffset int) (int, error) {
	if batchOffset == 0 {
		return 0, nil
	}
	if batchOffset == trainLen {
		return (trainLen + batchSize - 1) / batchSize, nil
	}
	if batchOffset%batchSize != 0 {
		return 0, fmt.Errorf("jevlike resumable batch offset %d is not a batch boundary", batchOffset)
	}
	return batchOffset / batchSize, nil
}

func frozenResumeEpochIndices(seed int64, epoch, n int) []int {
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	rng := rand.New(rand.NewSource(seed + int64(epoch)))
	rng.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})
	return indices
}

func frozenResumeNamedParametersToMap(params []NamedParameter) map[string][]float32 {
	out := make(map[string][]float32, len(params))
	for _, param := range params {
		out[param.Name] = append([]float32(nil), param.Values...)
	}
	return out
}

func frozenResumeAdamListToMap(list []frozenAdamState) map[string]adamState {
	out := make(map[string]adamState, len(list))
	for _, item := range list {
		out[item.Name] = adamState{M: append([]float32(nil), item.M...), V: append([]float32(nil), item.V...)}
	}
	return out
}

func frozenResumeAdamMapToList(states map[string]adamState, expected []NamedParameter) []frozenAdamState {
	out := make([]frozenAdamState, len(expected))
	for i, param := range expected {
		state := states[param.Name]
		m := append([]float32(nil), state.M...)
		v := append([]float32(nil), state.V...)
		if len(m) == 0 {
			m = make([]float32, len(param.Values))
		}
		if len(v) == 0 {
			v = make([]float32, len(param.Values))
		}
		out[i] = frozenAdamState{
			Name:  param.Name,
			Shape: append([]int(nil), param.Shape...),
			M:     m,
			V:     v,
		}
	}
	return out
}

func frozenResumeStateToResult(state frozenResumeState) TrainResult {
	result := TrainResult{
		History: append([]EpochMetrics(nil), state.History...),
	}
	if state.HasBest {
		result.BestValidationNLL = state.BestValidationNLL
		result.BestState = frozenResumeNamedParametersToMap(state.BestState)
	} else {
		result.BestValidationNLL = math.Inf(1)
	}
	return result
}
