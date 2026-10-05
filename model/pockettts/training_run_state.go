package pockettts

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"strconv"
)

const (
	MaxTrainingRunRows       = 100000
	MaxTrainingRunStateBytes = 2 << 20
	trainingRunAlgorithm     = "pcg-dxsm-fisher-yates-rejection-v1"
	trainingRunBoundary      = "completed-step-row1-constant-lr-dropout0-v1"
)

// TrainingRunIdentity is supplied explicitly by the caller, never inferred from
// globals. CorpusSHA256 must bind the manifest, split/cut/preprocessing policy
// and audio/latent provenance. OptimizerSHA256 must bind every AdamW/EMA setting
// and constant-LR policy; OptionsSHA256 binds other numerical/sampling options.
// V1 supports SIMD and explicitly named attention-backward, flow or transformer hybrids. Matching metadata does not prove the declared settings
// were used, or establish cross-toolchain/architecture float reproducibility.
type TrainingRunIdentity struct {
	SourceRevision     string `json:"source_revision"`
	GoVersion          string `json:"go_version"`
	GOOS               string `json:"goos"`
	GOARCH             string `json:"goarch"`
	SIMDMode           string `json:"simd_mode"`
	Backend            string `json:"backend"`
	CorpusSHA256       string `json:"corpus_sha256"`
	InitialModelSHA256 string `json:"initial_model_sha256"`
	TokenizerSHA256    string `json:"tokenizer_sha256"`
	MimiSHA256         string `json:"mimi_sha256"`
	ConfigSHA256       string `json:"config_sha256"`
	OptimizerSHA256    string `json:"optimizer_sha256"`
	OptionsSHA256      string `json:"options_sha256"`
}

// TrainingCheckpointBinding names one immutable generation file. The caller
// must hash that actual file and check its model step; this library never opens
// files. Independent sidecar/model writes are NOT crash-consistent publication.
type TrainingCheckpointBinding struct {
	Name   string `json:"name"`
	SHA256 string `json:"sha256"`
	Step   uint64 `json:"step"`
}

type trainingRunCore struct {
	identity                    TrainingRunIdentity
	rows                        int
	step, epoch                 uint64
	cursor                      int
	order                       []uint32
	orderRNG, timeRNG, noiseRNG [20]byte
}

// TrainingRunState is caller-owned, not concurrent-safe. V1 starts at optimiser
// step zero and accepts one row per completed update, no skips/accumulation.
// At epoch end cursor==rows is the normal form; reshuffling occurs only in a
// proposal for the next row, never on snapshot/decode.
type TrainingRunState struct {
	core trainingRunCore
}

// TrainingRowProposal owns an isolated next-step cursor/order/RNG state. A
// caller may abandon it without changing its parent. It contains no gradients
// or model state and cannot roll back a partially applied optimiser update.
type TrainingRowProposal struct {
	parent     *TrainingRunState
	parentStep uint64
	next       trainingRunCore
	row        uint32
	committed  bool
}

func NewTrainingRunState(identity TrainingRunIdentity, rows int, seeds [6]uint64) (*TrainingRunState, error) {
	if err := identity.validate(); err != nil {
		return nil, err
	}
	if rows <= 0 || rows > MaxTrainingRunRows {
		return nil, fmt.Errorf("training run row bound exceeded")
	}
	c := trainingRunCore{identity: identity, rows: rows, order: make([]uint32, rows)}
	c.orderRNG = pcgState(rand.NewPCG(seeds[0], seeds[1]))
	c.timeRNG = pcgState(rand.NewPCG(seeds[2], seeds[3]))
	c.noiseRNG = pcgState(rand.NewPCG(seeds[4], seeds[5]))
	shuffleRunOrder(&c)
	return &TrainingRunState{core: c}, nil
}

func (s *TrainingRunState) Progress() (step, epoch uint64, cursor int) {
	if s == nil {
		return 0, 0, 0
	}
	return s.core.step, s.core.epoch, s.core.cursor
}
func (s *TrainingRunState) RowOrder() []uint32 {
	if s == nil {
		return nil
	}
	return append([]uint32(nil), s.core.order...)
}
func (s *TrainingRunState) PrepareRow() (*TrainingRowProposal, error) {
	if s == nil {
		return nil, fmt.Errorf("nil training run state")
	}
	if err := s.core.validate(); err != nil {
		return nil, err
	}
	if s.core.step == math.MaxUint64 {
		return nil, fmt.Errorf("training run step overflow")
	}
	next := s.core.clone()
	if next.cursor == next.rows {
		next.epoch++ // bounded by next.step, already checked for overflow
		next.cursor = 0
		shuffleRunOrder(&next)
	}
	row := next.order[next.cursor]
	next.cursor++
	next.step++
	return &TrainingRowProposal{parent: s, parentStep: s.core.step, next: next, row: row}, nil
}
func (p *TrainingRowProposal) Row() (uint32, error) {
	if p == nil || p.committed {
		return 0, fmt.Errorf("inactive training row proposal")
	}
	return p.row, nil
}
func (p *TrainingRowProposal) TimeUint64() (uint64, error) {
	if p == nil || p.committed {
		return 0, fmt.Errorf("inactive training row proposal")
	}
	return drawPCG(&p.next.timeRNG), nil
}
func (p *TrainingRowProposal) NoiseUint64() (uint64, error) {
	if p == nil || p.committed {
		return 0, fmt.Errorf("inactive training row proposal")
	}
	return drawPCG(&p.next.noiseRNG), nil
}

// CommitRow accepts only declared evidence of the next completed optimiser+EMA
// step. It cannot verify that update succeeded. After a failed/uncertain model
// update, discard the proposal AND recover a coherent prior model checkpoint;
// do not blindly replay draws against partially updated parameters.
func (s *TrainingRunState) CommitRow(p *TrainingRowProposal, completedStep uint64) error {
	if s == nil || p == nil || p.parent != s || p.committed || p.parentStep != s.core.step {
		return fmt.Errorf("stale/replayed/wrong-parent training proposal")
	}
	if s.core.step == math.MaxUint64 || completedStep != s.core.step+1 || p.next.step != completedStep {
		return fmt.Errorf("completed optimiser step must advance exactly one row")
	}
	if p.next.identity != s.core.identity || p.next.rows != s.core.rows {
		return fmt.Errorf("training proposal identity changed")
	}
	if err := p.next.validate(); err != nil {
		return err
	}
	s.core = p.next.clone()
	p.committed = true
	return nil
}

type trainingRunWire struct {
	Version    uint64                    `json:"version"`
	Algorithm  string                    `json:"algorithm"`
	Boundary   string                    `json:"boundary"`
	Identity   TrainingRunIdentity       `json:"identity"`
	Checkpoint TrainingCheckpointBinding `json:"checkpoint"`
	Rows       int                       `json:"rows"`
	Step       uint64                    `json:"step"`
	Epoch      uint64                    `json:"epoch"`
	Cursor     int                       `json:"cursor"`
	Order      []uint32                  `json:"order"`
	OrderRNG   string                    `json:"order_rng"`
	TimeRNG    string                    `json:"time_rng"`
	NoiseRNG   string                    `json:"noise_rng"`
}

// Snapshot binds a COPY of current mutable state to an explicitly supplied
// completed checkpoint. The live state never retains an old checkpoint digest.
func (s *TrainingRunState) Snapshot(binding TrainingCheckpointBinding) ([]byte, error) {
	if s == nil {
		return nil, fmt.Errorf("nil training run state")
	}
	if err := s.core.validate(); err != nil {
		return nil, err
	}
	if err := binding.validate(); err != nil {
		return nil, err
	}
	if binding.Step != s.core.step {
		return nil, fmt.Errorf("checkpoint step differs from run step")
	}
	c := s.core
	w := trainingRunWire{1, trainingRunAlgorithm, trainingRunBoundary, c.identity, binding, c.rows, c.step, c.epoch, c.cursor, c.order, hex.EncodeToString(c.orderRNG[:]), hex.EncodeToString(c.timeRNG[:]), hex.EncodeToString(c.noiseRNG[:])}
	data, err := json.Marshal(w)
	if err != nil {
		return nil, err
	}
	if len(data) > MaxTrainingRunStateBytes {
		return nil, fmt.Errorf("training run snapshot byte limit exceeded")
	}
	return data, nil
}

// RestoreTrainingRunState rejects mismatched declared identity/checkpoint before
// exposing new state, performs no implicit draw/reshuffle and mutates no model.
// It validates declarations, not on-disk content or two-file crash durability.
func RestoreTrainingRunState(data []byte, expected TrainingRunIdentity, checkpoint TrainingCheckpointBinding) (*TrainingRunState, error) {
	if err := expected.validate(); err != nil {
		return nil, err
	}
	if err := checkpoint.validate(); err != nil {
		return nil, err
	}
	if len(data) == 0 || len(data) > MaxTrainingRunStateBytes {
		return nil, fmt.Errorf("training run snapshot byte limit exceeded")
	}
	fields, err := runJSONObject(data, []string{"version", "algorithm", "boundary", "identity", "checkpoint", "rows", "step", "epoch", "cursor", "order", "order_rng", "time_rng", "noise_rng"})
	if err != nil {
		return nil, err
	}
	version, err := runUint(fields["version"])
	if err != nil || version != 1 {
		return nil, fmt.Errorf("unsupported training run schema")
	}
	algorithm, err := runString(fields["algorithm"], 64)
	if err != nil || algorithm != trainingRunAlgorithm {
		return nil, fmt.Errorf("training run RNG/shuffle algorithm mismatch")
	}
	boundary, err := runString(fields["boundary"], 64)
	if err != nil || boundary != trainingRunBoundary {
		return nil, fmt.Errorf("unsupported accumulation/schedule/update boundary")
	}
	identity, err := decodeRunIdentity(fields["identity"])
	if err != nil {
		return nil, err
	}
	if identity != expected {
		return nil, fmt.Errorf("training run numerical/input/optimizer identity mismatch")
	}
	binding, err := decodeRunBinding(fields["checkpoint"])
	if err != nil {
		return nil, err
	}
	if binding != checkpoint {
		return nil, fmt.Errorf("training run checkpoint binding mismatch")
	}
	rows, err := runUint(fields["rows"])
	if err != nil || rows == 0 || rows > MaxTrainingRunRows {
		return nil, fmt.Errorf("training run row bound exceeded")
	}
	step, err := runUint(fields["step"])
	if err != nil {
		return nil, err
	}
	if step != checkpoint.Step {
		return nil, fmt.Errorf("checkpoint step differs from run step")
	}
	epoch, err := runUint(fields["epoch"])
	if err != nil {
		return nil, err
	}
	cursor, err := runUint(fields["cursor"])
	if err != nil || cursor > rows {
		return nil, fmt.Errorf("invalid training run cursor")
	}
	order, err := decodeRunOrder(fields["order"], int(rows))
	if err != nil {
		return nil, err
	}
	c := trainingRunCore{identity: identity, rows: int(rows), step: step, epoch: epoch, cursor: int(cursor), order: order}
	for _, pair := range []struct {
		key string
		dst *[20]byte
	}{{"order_rng", &c.orderRNG}, {"time_rng", &c.timeRNG}, {"noise_rng", &c.noiseRNG}} {
		text, err := runString(fields[pair.key], 40)
		if err != nil {
			return nil, err
		}
		if len(text) != 40 {
			return nil, fmt.Errorf("invalid RNG state length")
		}
		raw, err := hex.DecodeString(text)
		if err != nil || hex.EncodeToString(raw) != text {
			return nil, fmt.Errorf("invalid RNG state hex")
		}
		copy(pair.dst[:], raw)
	}
	if err := c.validate(); err != nil {
		return nil, err
	}
	return &TrainingRunState{core: c}, nil
}

func (c trainingRunCore) clone() trainingRunCore {
	c.order = append([]uint32(nil), c.order...)
	return c
}
func (c trainingRunCore) validate() error {
	if err := c.identity.validate(); err != nil {
		return err
	}
	if c.rows <= 0 || c.rows > MaxTrainingRunRows || len(c.order) != c.rows || c.cursor < 0 || c.cursor > c.rows {
		return fmt.Errorf("invalid training run permutation/cursor bound")
	}
	if c.step == 0 {
		if c.epoch != 0 || c.cursor != 0 {
			return fmt.Errorf("step-zero cursor/epoch mismatch")
		}
	} else if c.epoch != (c.step-1)/uint64(c.rows) || c.cursor != int((c.step-1)%uint64(c.rows))+1 {
		return fmt.Errorf("epoch/cursor/completed-step mismatch")
	}
	seen := make([]bool, c.rows)
	for _, row := range c.order {
		if uint64(row) >= uint64(c.rows) || seen[row] {
			return fmt.Errorf("invalid/duplicate training row permutation")
		}
		seen[row] = true
	}
	for _, raw := range [][20]byte{c.orderRNG, c.timeRNG, c.noiseRNG} {
		var p rand.PCG
		if err := p.UnmarshalBinary(raw[:]); err != nil {
			return fmt.Errorf("invalid PCG state: %w", err)
		}
	}
	return nil
}
func (i TrainingRunIdentity) validate() error {
	if !runHex(i.SourceRevision, 40) || (i.Backend != "simd" && i.Backend != "ptx-attention-hybrid-v1" && i.Backend != "vulkan-attention-hybrid-v1" && i.Backend != "ptx-flow-hybrid-v1" && i.Backend != "vulkan-flow-hybrid-v1" && i.Backend != "ptx-transformer-hybrid-v1" && i.Backend != "vulkan-transformer-hybrid-v1" && i.Backend != "ptx-optimizer-hybrid-v1" && i.Backend != "vulkan-optimizer-hybrid-v1" && i.Backend != "ptx-resident-hybrid-v1" && i.Backend != "vulkan-resident-hybrid-v1" && i.Backend != "ptx-resident-weight-hybrid-v1" && i.Backend != "vulkan-resident-weight-hybrid-v1" && i.Backend != "ptx-resident-projection-hybrid-v1" && i.Backend != "vulkan-resident-projection-hybrid-v1" && i.Backend != "ptx-resident-sequence-hybrid-v1" && i.Backend != "vulkan-resident-sequence-hybrid-v1" && i.Backend != "ptx-resident-time-hybrid-v1" && i.Backend != "vulkan-resident-time-hybrid-v1" && i.Backend != "ptx-resident-eos-hybrid-v1" && i.Backend != "vulkan-resident-eos-hybrid-v1" && i.Backend != "ptx-resident-loss-hybrid-v1" && i.Backend != "vulkan-resident-loss-hybrid-v1" && i.Backend != "ptx-resident-branch-hybrid-v1" && i.Backend != "vulkan-resident-branch-hybrid-v1" && i.Backend != "ptx-resident-step-hybrid-v1" && i.Backend != "vulkan-resident-step-hybrid-v1" && i.Backend != "ptx-resident-pruned-hybrid-v1" && i.Backend != "vulkan-resident-pruned-hybrid-v1" && i.Backend != "ptx-resident-globalnorm-hybrid-v1" && i.Backend != "ptx-resident-devicescale-hybrid-v1" && i.Backend != "ptx-resident-sampleweight-hybrid-v1" && i.Backend != "ptx-resident-lifetime-hybrid-v1" && i.Backend != "ptx-resident-statecheck-hybrid-v1" && i.Backend != "ptx-resident-zeroconst-hybrid-v1" && i.Backend != "ptx-resident-eosfused-hybrid-v1" && i.Backend != "ptx-resident-audiofused-hybrid-v1" && i.Backend != "ptx-resident-voicefused-hybrid-v1" && i.Backend != "ptx-resident-conditioner-hybrid-v1" && i.Backend != "ptx-resident-backbone-hybrid-v1" && i.Backend != "ptx-resident-eosbackbone-hybrid-v1" && i.Backend != "ptx-resident-diagfused-hybrid-v1" && i.Backend != "ptx-resident-endpointfused-hybrid-v1" && i.Backend != "ptx-resident-primaryfused-hybrid-v1" && i.Backend != "ptx-resident-distillfused-hybrid-v1" && i.Backend != "ptx-resident-forwardfused-hybrid-v1" && i.Backend != "ptx-resident-diagforward-hybrid-v1" && i.Backend != "ptx-resident-weightroots-hybrid-v1" && i.Backend != "ptx-resident-stepweight-hybrid-v1" && i.Backend != "ptx-resident-stepbackbone-hybrid-v1" && i.Backend != "ptx-resident-stepresults-hybrid-v1" && i.Backend != "ptx-resident-distillresults-hybrid-v1" && i.Backend != "ptx-resident-eosforward-hybrid-v1" && i.Backend != "ptx-resident-forwardresults-hybrid-v1" && i.Backend != "ptx-resident-embeddingindex-hybrid-v1" && i.Backend != "ptx-resident-embeddingcompact-hybrid-v1" && i.Backend != "ptx-resident-diagadmission-hybrid-v1" && i.Backend != "ptx-resident-distilladmission-hybrid-v1" && i.Backend != "ptx-resident-flowhandoff-hybrid-v1" && i.Backend != "ptx-resident-eosadmission-hybrid-v1" && i.Backend != "ptx-resident-devicehandoff-hybrid-v1" && i.Backend != "ptx-resident-predictionhandoff-hybrid-v1" && i.Backend != "vulkan-resident-globalnorm-hybrid-v1" && i.Backend != "vulkan-resident-devicescale-hybrid-v1" && i.Backend != "vulkan-resident-sampleweight-hybrid-v1" && i.Backend != "vulkan-resident-lifetime-hybrid-v1" && i.Backend != "vulkan-resident-statecheck-hybrid-v1" && i.Backend != "vulkan-resident-zeroconst-hybrid-v1" && i.Backend != "vulkan-resident-eosfused-hybrid-v1" && i.Backend != "vulkan-resident-audiofused-hybrid-v1" && i.Backend != "vulkan-resident-voicefused-hybrid-v1" && i.Backend != "vulkan-resident-conditioner-hybrid-v1" && i.Backend != "vulkan-resident-backbone-hybrid-v1" && i.Backend != "vulkan-resident-eosbackbone-hybrid-v1" && i.Backend != "vulkan-resident-diagfused-hybrid-v1" && i.Backend != "vulkan-resident-endpointfused-hybrid-v1" && i.Backend != "vulkan-resident-primaryfused-hybrid-v1" && i.Backend != "vulkan-resident-distillfused-hybrid-v1" && i.Backend != "vulkan-resident-forwardfused-hybrid-v1" && i.Backend != "vulkan-resident-diagforward-hybrid-v1" && i.Backend != "vulkan-resident-weightroots-hybrid-v1" && i.Backend != "vulkan-resident-stepweight-hybrid-v1" && i.Backend != "vulkan-resident-stepbackbone-hybrid-v1" && i.Backend != "vulkan-resident-stepresults-hybrid-v1" && i.Backend != "vulkan-resident-distillresults-hybrid-v1" && i.Backend != "vulkan-resident-eosforward-hybrid-v1" && i.Backend != "vulkan-resident-forwardresults-hybrid-v1" && i.Backend != "vulkan-resident-embeddingindex-hybrid-v1" && i.Backend != "vulkan-resident-embeddingcompact-hybrid-v1" && i.Backend != "vulkan-resident-diagadmission-hybrid-v1" && i.Backend != "vulkan-resident-distilladmission-hybrid-v1" && i.Backend != "vulkan-resident-flowhandoff-hybrid-v1" && i.Backend != "vulkan-resident-eosadmission-hybrid-v1" && i.Backend != "vulkan-resident-devicehandoff-hybrid-v1" && i.Backend != "vulkan-resident-predictionhandoff-hybrid-v1" && i.Backend != "vulkan-resident-sharedworkspace-hybrid-v1") {
		return fmt.Errorf("training run requires pinned source and supported explicit backend scope")
	}
	for _, s := range []string{i.GoVersion, i.GOOS, i.GOARCH, i.SIMDMode} {
		if len(s) == 0 || len(s) > 64 {
			return fmt.Errorf("missing/oversized numerical identity")
		}
		for _, r := range s {
			if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '.' || r == '-' || r == '_') {
				return fmt.Errorf("invalid numerical identity syntax")
			}
		}
	}
	for _, s := range []string{i.CorpusSHA256, i.InitialModelSHA256, i.TokenizerSHA256, i.MimiSHA256, i.ConfigSHA256, i.OptimizerSHA256, i.OptionsSHA256} {
		if !runHex(s, 64) {
			return fmt.Errorf("invalid training run input/options digest")
		}
	}
	return nil
}
func (b TrainingCheckpointBinding) validate() error {
	if !runHex(b.SHA256, 64) || len(b.Name) == 0 || len(b.Name) > 128 || b.Name == "." || b.Name == ".." {
		return fmt.Errorf("invalid checkpoint generation binding")
	}
	for _, r := range b.Name {
		if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '.' || r == '-' || r == '_') {
			return fmt.Errorf("checkpoint must be a confined relative generation filename")
		}
	}
	return nil
}
func runHex(s string, length int) bool {
	if len(s) != length {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}
func pcgState(p *rand.PCG) [20]byte {
	var out [20]byte
	raw, _ := p.MarshalBinary()
	copy(out[:], raw)
	return out
}
func drawPCG(state *[20]byte) uint64 {
	var p rand.PCG
	_ = p.UnmarshalBinary(state[:])
	v := p.Uint64()
	*state = pcgState(&p)
	return v
}
func shuffleRunOrder(c *trainingRunCore) {
	for i := range c.order {
		c.order[i] = uint32(i)
	}
	for i := len(c.order) - 1; i > 0; i-- {
		n := uint64(i + 1)
		threshold := -n % n
		var v uint64
		for {
			v = drawPCG(&c.orderRNG)
			if v >= threshold {
				break
			}
		}
		j := int(v % n)
		c.order[i], c.order[j] = c.order[j], c.order[i]
	}
}

// All raw JSON values are bounded before typed allocation; duplicate/unknown/
// missing/null keys are rejected. No general parser or filesystem API is added.
func runJSONObject(data []byte, keys []string) (map[string]json.RawMessage, error) {
	d := json.NewDecoder(bytes.NewReader(data))
	first, err := d.Token()
	if err != nil || first != json.Delim('{') {
		return nil, fmt.Errorf("expected non-null run-state object")
	}
	out := make(map[string]json.RawMessage, len(keys))
	allowed := make(map[string]bool, len(keys))
	for _, k := range keys {
		allowed[k] = true
	}
	for d.More() {
		key, err := d.Token()
		if err != nil {
			return nil, err
		}
		k, ok := key.(string)
		if !ok || !allowed[k] {
			return nil, fmt.Errorf("unknown run-state field")
		}
		if _, ok := out[k]; ok {
			return nil, fmt.Errorf("duplicate run-state field %q", k)
		}
		var v json.RawMessage
		if err := d.Decode(&v); err != nil {
			return nil, err
		}
		if bytes.Equal(bytes.TrimSpace(v), []byte("null")) {
			return nil, fmt.Errorf("null run-state field %q", k)
		}
		out[k] = v
	}
	if _, err := d.Token(); err != nil {
		return nil, err
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return nil, fmt.Errorf("trailing run-state data")
	}
	if len(out) != len(keys) {
		return nil, fmt.Errorf("missing run-state field")
	}
	return out, nil
}
func runUint(raw []byte) (uint64, error) {
	s := string(bytes.TrimSpace(raw))
	if len(s) == 0 {
		return 0, fmt.Errorf("expected unsigned integer")
	}
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0, fmt.Errorf("expected unsigned integer syntax")
		}
	}
	return strconv.ParseUint(s, 10, 64)
}
func runString(raw []byte, limit int) (string, error) {
	if len(raw) > limit*6+2 {
		return "", fmt.Errorf("run-state string byte bound exceeded")
	}
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || raw[0] != '"' {
		return "", fmt.Errorf("expected run-state string")
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return "", err
	}
	if len(s) > limit {
		return "", fmt.Errorf("run-state string bound exceeded")
	}
	return s, nil
}
func decodeRunOrder(raw []byte, rows int) ([]uint32, error) {
	d := json.NewDecoder(bytes.NewReader(raw))
	first, err := d.Token()
	if err != nil || first != json.Delim('[') {
		return nil, fmt.Errorf("expected run-state permutation array")
	}
	out := make([]uint32, 0, rows)
	for d.More() {
		if len(out) == rows {
			return nil, fmt.Errorf("permutation length exceeds admitted rows")
		}
		var value json.RawMessage
		if err := d.Decode(&value); err != nil {
			return nil, err
		}
		n, err := runUint(value)
		if err != nil || n >= uint64(rows) {
			return nil, fmt.Errorf("invalid permutation row")
		}
		out = append(out, uint32(n))
	}
	if _, err := d.Token(); err != nil {
		return nil, err
	}
	if len(out) != rows {
		return nil, fmt.Errorf("permutation length mismatch")
	}
	return out, nil
}
func decodeRunIdentity(raw []byte) (TrainingRunIdentity, error) {
	var i TrainingRunIdentity
	keys := []string{"source_revision", "go_version", "goos", "goarch", "simd_mode", "backend", "corpus_sha256", "initial_model_sha256", "tokenizer_sha256", "mimi_sha256", "config_sha256", "optimizer_sha256", "options_sha256"}
	fields, err := runJSONObject(raw, keys)
	if err != nil {
		return i, err
	}
	values := []*string{&i.SourceRevision, &i.GoVersion, &i.GOOS, &i.GOARCH, &i.SIMDMode, &i.Backend, &i.CorpusSHA256, &i.InitialModelSHA256, &i.TokenizerSHA256, &i.MimiSHA256, &i.ConfigSHA256, &i.OptimizerSHA256, &i.OptionsSHA256}
	for n, k := range keys {
		s, err := runString(fields[k], 64)
		if err != nil {
			return i, err
		}
		*values[n] = s
	}
	return i, i.validate()
}
func decodeRunBinding(raw []byte) (TrainingCheckpointBinding, error) {
	var b TrainingCheckpointBinding
	fields, err := runJSONObject(raw, []string{"name", "sha256", "step"})
	if err != nil {
		return b, err
	}
	b.Name, err = runString(fields["name"], 128)
	if err != nil {
		return b, err
	}
	b.SHA256, err = runString(fields["sha256"], 64)
	if err != nil {
		return b, err
	}
	b.Step, err = runUint(fields["step"])
	if err != nil {
		return b, err
	}
	return b, b.validate()
}
