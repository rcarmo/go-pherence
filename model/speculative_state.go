package model

import (
	"fmt"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

// CPUDecodeState is the incremental state shape needed by a KV-reusing
// speculative verifier. The current generator still owns the full token-step
// implementation, but keeping this state contract explicit lets the verifier
// block land without changing the public GenerateSpeculative API again.
type CPUDecodeState struct {
	Model          *LlamaModel
	Output         []int
	KVCacheK       [][]float32
	KVCacheV       [][]float32
	CompressedKV   []*kv.CompressedKVCache
	LayeredKV      *kv.LayeredF32KV
	KVDims         []int
	backend        string
	layeredKVIndex []int
}

// CPUDecodeCheckpoint records a restorable point before staging verifier tokens.
type CPUDecodeCheckpoint struct {
	OutputLen    int
	FloatKV      kv.FloatKVCheckpoint
	CompressedKV []kv.CompressedKVCheckpoint
	LayeredKV    kv.LayeredF32KVCheckpoint
}

func NewCPUDecodeStateForSpeculative(m *LlamaModel, prepared []int, maxTokens int, backend ...string) (*CPUDecodeState, error) {
	if m == nil {
		return nil, fmt.Errorf("nil model")
	}
	if maxTokens < 0 {
		return nil, fmt.Errorf("maxTokens=%d must be >= 0", maxTokens)
	}
	dims, err := m.LayerKVDims()
	if err != nil {
		return nil, err
	}
	selectedBackend := "replay"
	if len(backend) > 0 && backend[0] != "" {
		selectedBackend = backend[0]
	}
	if selectedBackend != "replay" {
		selectedBackend = "replay"
	}
	st := &CPUDecodeState{
		Model:    m,
		Output:   append([]int(nil), prepared...),
		KVCacheK: make([][]float32, len(m.Layers)),
		KVCacheV: make([][]float32, len(m.Layers)),
		KVDims:   dims,
		backend:  selectedBackend,
	}
	maxInt := int(^uint(0) >> 1)
	if maxTokens > maxInt-len(prepared) {
		return nil, fmt.Errorf("KV max sequence overflow: prepared=%d maxTokens=%d", len(prepared), maxTokens)
	}
	maxSeq := len(prepared) + maxTokens
	if maxSeq < 1 {
		maxSeq = 1
	}
	for i, dim := range dims {
		if dim <= 0 {
			continue
		}
		if maxSeq > maxInt/dim {
			return nil, fmt.Errorf("layer %d KV capacity overflow: seq=%d dim=%d", i, maxSeq, dim)
		}
		st.KVCacheK[i] = make([]float32, 0, maxSeq*dim)
		st.KVCacheV[i] = make([]float32, 0, maxSeq*dim)
	}
	return st, nil
}

func (s *CPUDecodeState) VerifierBackend() string {
	if s == nil || s.backend == "" {
		return "replay"
	}
	return s.backend
}

func (s *CPUDecodeState) GenerateGreedy(n int) error {
	if n < 0 {
		return fmt.Errorf("generate greedy n=%d must be >= 0", n)
	}
	for i := 0; i < n; i++ {
		if _, err := s.DecodeOneGreedy(); err != nil {
			return err
		}
	}
	return nil
}

func (s *CPUDecodeState) DecodeOneGreedy() (int, error) {
	if s == nil || s.Model == nil {
		return 0, fmt.Errorf("nil decode state/model")
	}
	verified := s.Model.generatePrepared(s.Output, 1)
	if len(verified) <= len(s.Output) {
		return 0, fmt.Errorf("decode produced no token")
	}
	tok := verified[len(s.Output)]
	s.Output = append(s.Output, tok)
	return tok, nil
}

// EnableLayeredF32KV migrates the state's current float32 K/V slices into a
// mixed full-history/sliding-window layered manager and clears the slice-backed
// caches after validating the seeded rows.
func (s *CPUDecodeState) EnableLayeredF32KV(maxPrefillChunk int) error {
	if s == nil || s.Model == nil {
		return fmt.Errorf("nil decode state/model")
	}
	if s.LayeredKV != nil {
		return fmt.Errorf("layered KV already enabled")
	}
	dims, configs, layeredIndex, err := s.layeredKVConfig()
	if err != nil {
		return err
	}
	if len(s.KVCacheK) != len(dims) || len(s.KVCacheV) != len(dims) {
		return fmt.Errorf("layered KV seed layers K/V=%d/%d, want %d", len(s.KVCacheK), len(s.KVCacheV), len(dims))
	}
	manager, err := kv.NewLayeredF32KV(configs, maxPrefillChunk)
	if err != nil {
		return err
	}
	seqLen := len(s.Output)
	for layer, dim := range dims {
		rows, err := validateLayeredSeedRows(layer, dim, seqLen, s.KVCacheK[layer], s.KVCacheV[layer])
		if err != nil {
			return err
		}
		idx := layeredIndex[layer]
		if idx < 0 {
			continue
		}
		for row := 0; row < rows; row++ {
			off := row * dim
			if err := manager.Append(idx, s.KVCacheK[layer][off:off+dim], s.KVCacheV[layer][off:off+dim]); err != nil {
				return fmt.Errorf("layered KV seed layer %d row %d: %w", layer, row, err)
			}
		}
		if err := validateLayeredSeedMaterialization(manager, idx, layer, dim, s.KVCacheK[layer], s.KVCacheV[layer]); err != nil {
			return err
		}
	}
	s.LayeredKV = manager
	s.layeredKVIndex = layeredIndex
	s.KVDims = dims
	for i := range s.KVCacheK {
		s.KVCacheK[i] = nil
		s.KVCacheV[i] = nil
	}
	return nil
}

// MaterializeLayeredKV expands the layered manager into absolute-position-aligned
// float32 K/V slices for existing verifier seams. Sliding-window layers leave
// evicted prefixes as zero rows because current verifier paths only index the
// currently addressable suffix.
func (s *CPUDecodeState) MaterializeLayeredKV() ([][]float32, [][]float32, error) {
	if s == nil || s.Model == nil {
		return nil, nil, fmt.Errorf("nil decode state/model")
	}
	if s.LayeredKV == nil {
		return nil, nil, fmt.Errorf("layered KV is not enabled")
	}
	if len(s.KVDims) != len(s.Model.Layers) {
		return nil, nil, fmt.Errorf("layered KV dims=%d, want layers=%d", len(s.KVDims), len(s.Model.Layers))
	}
	layeredIndex := s.layeredKVIndex
	if len(layeredIndex) != len(s.Model.Layers) {
		_, _, layeredIndex, err := s.layeredKVConfig()
		if err != nil {
			return nil, nil, err
		}
		s.layeredKVIndex = layeredIndex
	}
	seqLen := len(s.Output)
	k := make([][]float32, len(s.Model.Layers))
	v := make([][]float32, len(s.Model.Layers))
	for layer, dim := range s.KVDims {
		if dim <= 0 {
			continue
		}
		idx := layeredIndex[layer]
		if idx < 0 {
			return nil, nil, fmt.Errorf("layered KV layer %d missing manager index", layer)
		}
		mk, mv, startToken, err := s.LayeredKV.MaterializeLayer(idx)
		if err != nil {
			return nil, nil, fmt.Errorf("layered KV layer %d: %w", layer, err)
		}
		if len(mk) != len(mv) || len(mk)%dim != 0 {
			return nil, nil, fmt.Errorf("layered KV layer %d materialized K/V=%d/%d incompatible with dim=%d", layer, len(mk), len(mv), dim)
		}
		rows := len(mk) / dim
		if startToken < 0 || startToken > seqLen {
			return nil, nil, fmt.Errorf("layered KV layer %d start token=%d outside seq len=%d", layer, startToken, seqLen)
		}
		if rows != seqLen-startToken {
			return nil, nil, fmt.Errorf("layered KV layer %d rows=%d, want suffix len=%d from start=%d seq=%d", layer, rows, seqLen-startToken, startToken, seqLen)
		}
		need, ok := checkedProduct(seqLen, dim)
		if !ok {
			return nil, nil, fmt.Errorf("layered KV layer %d materialized length overflows seq=%d dim=%d", layer, seqLen, dim)
		}
		prefix, ok := checkedProduct(startToken, dim)
		if !ok {
			return nil, nil, fmt.Errorf("layered KV layer %d prefix offset overflows start=%d dim=%d", layer, startToken, dim)
		}
		k[layer] = make([]float32, need)
		v[layer] = make([]float32, need)
		copy(k[layer][prefix:], mk)
		copy(v[layer][prefix:], mv)
	}
	return k, v, nil
}

func (s *CPUDecodeState) Checkpoint() CPUDecodeCheckpoint {
	cp := CPUDecodeCheckpoint{OutputLen: len(s.Output)}
	if s.LayeredKV != nil {
		cp.LayeredKV = s.LayeredKV.Checkpoint()
	} else if s.CompressedKV != nil {
		cp.CompressedKV = kv.CheckpointCompressedKV(s.CompressedKV)
	} else {
		cp.FloatKV = kv.CheckpointFloatKV(s.KVCacheK, s.KVCacheV)
	}
	return cp
}

func (s *CPUDecodeState) Restore(cp CPUDecodeCheckpoint) error {
	if cp.OutputLen < 0 || cp.OutputLen > len(s.Output) {
		return fmt.Errorf("checkpoint output len=%d outside current len=%d", cp.OutputLen, len(s.Output))
	}
	s.Output = s.Output[:cp.OutputLen]
	if s.LayeredKV != nil {
		return s.LayeredKV.Restore(cp.LayeredKV)
	}
	if s.CompressedKV != nil {
		return kv.RestoreCompressedKV(s.CompressedKV, cp.CompressedKV)
	}
	return cp.FloatKV.Restore(s.KVCacheK, s.KVCacheV)
}

func (s *CPUDecodeState) CommitAcceptedOutputOnly(cp CPUDecodeCheckpoint, acceptance MTPAcceptance) error {
	if err := acceptance.Validate(); err != nil {
		return err
	}
	if cp.OutputLen < 0 || cp.OutputLen > len(s.Output) {
		return fmt.Errorf("checkpoint output len=%d outside current len=%d", cp.OutputLen, len(s.Output))
	}
	s.Output = s.Output[:cp.OutputLen]
	s.Output = append(s.Output, acceptance.OutputTokens...)
	return nil
}

// VerifyGreedyBlock verifies drafted tokens with the real model and returns the
// greedy acceptance result. This first implementation intentionally uses the
// existing prepared-prompt CPU generator as the verifier backend; replacing the
// body with a KV-reusing DecodeOne loop should not require changes in
// GenerateSpeculative.
func (s *CPUDecodeState) VerifyGreedyBlock(drafted []int) (MTPAcceptance, error) {
	if s == nil || s.Model == nil {
		return MTPAcceptance{}, fmt.Errorf("nil decode state/model")
	}
	verifyN := len(drafted) + 1
	shadow := *s
	shadow.Output = append([]int(nil), s.Output...)
	verifierTokens := make([]int, 0, verifyN)
	for i := 0; i < verifyN; i++ {
		tok, err := shadow.DecodeOneGreedy()
		if err != nil {
			return MTPAcceptance{}, fmt.Errorf("verifier token %d: %w", i, err)
		}
		verifierTokens = append(verifierTokens, tok)
	}
	return AcceptMTPDraft(drafted, verifierTokens)
}

func (s *CPUDecodeState) CommitAccepted(cp CPUDecodeCheckpoint, acceptance MTPAcceptance) error {
	if err := acceptance.Validate(); err != nil {
		return err
	}
	if cp.OutputLen < 0 || cp.OutputLen > len(s.Output) {
		return fmt.Errorf("checkpoint output len=%d outside current len=%d", cp.OutputLen, len(s.Output))
	}
	if s.LayeredKV != nil {
		if err := commitAcceptedLayeredKV(s.LayeredKV, cp.LayeredKV, acceptance); err != nil {
			return err
		}
	} else if s.CompressedKV != nil {
		if err := kv.KeepCompressedKVAppended(s.CompressedKV, cp.CompressedKV, acceptance.KVKeepTokens()); err != nil {
			return err
		}
	} else {
		if err := CommitAcceptedFloatKV(s.KVCacheK, s.KVCacheV, cp.FloatKV, s.KVDims, acceptance); err != nil {
			return err
		}
	}
	s.Output = s.Output[:cp.OutputLen]
	s.Output = append(s.Output, acceptance.OutputTokens...)
	return nil
}

// CommitGraphAccepted is the production-facing MTP commit bridge: a verifier
// result must match the explicit execution graph before accepted-prefix+bonus KV
// and output tokens are retained. This is the path future Gemma4 MTP generation
// should use instead of committing a bare MTPAcceptance.
func (s *CPUDecodeState) CommitGraphAccepted(cp CPUDecodeCheckpoint, graph MTPExecutionGraph, verifier MTPVerifierResult) (MTPKVCommitPlan, error) {
	if s == nil || s.Model == nil {
		return MTPKVCommitPlan{}, fmt.Errorf("nil decode state/model")
	}
	if cp.OutputLen < 0 || cp.OutputLen > len(s.Output) {
		return MTPKVCommitPlan{}, fmt.Errorf("checkpoint output len=%d outside current len=%d", cp.OutputLen, len(s.Output))
	}
	if cp.OutputLen != graph.StartPos {
		return MTPKVCommitPlan{}, fmt.Errorf("checkpoint output len=%d does not match MTP graph start position=%d", cp.OutputLen, graph.StartPos)
	}
	var commit MTPKVCommitPlan
	var err error
	if s.LayeredKV != nil {
		if err := verifier.validateGraphForModel(s.Model, graph); err != nil {
			return MTPKVCommitPlan{}, err
		}
		commit, err = graph.CommitPlan(verifier.Acceptance)
		if err != nil {
			return MTPKVCommitPlan{}, err
		}
		if err := commitAcceptedLayeredKV(s.LayeredKV, cp.LayeredKV, verifier.Acceptance); err != nil {
			return MTPKVCommitPlan{}, err
		}
	} else if s.CompressedKV != nil {
		commit, err = verifier.CommitGraphCompressedKVForModel(s.Model, graph, s.CompressedKV, cp.CompressedKV)
	} else {
		commit, err = verifier.CommitGraphFloatKV(s.Model, graph, s.KVCacheK, s.KVCacheV, cp.FloatKV)
	}
	if err != nil {
		return MTPKVCommitPlan{}, err
	}
	s.Output = s.Output[:cp.OutputLen]
	s.Output = append(s.Output, commit.OutputTokens...)
	return commit, nil
}

func (s *CPUDecodeState) layeredKVConfig() ([]int, []kv.LayerF32KVConfig, []int, error) {
	if s == nil || s.Model == nil {
		return nil, nil, nil, fmt.Errorf("nil decode state/model")
	}
	dims, err := s.Model.LayerKVDims()
	if err != nil {
		return nil, nil, nil, err
	}
	configs := make([]kv.LayerF32KVConfig, 0, len(dims))
	layeredIndex := make([]int, len(dims))
	for i := range layeredIndex {
		layeredIndex[i] = -1
	}
	for layer, dim := range dims {
		if dim <= 0 {
			continue
		}
		cfg := kv.LayerF32KVConfig{Dim: dim}
		if s.Model.Config.SlidingWindow > 0 && len(s.Model.Config.LayerTypes) > layer && s.Model.Config.LayerTypes[layer] == "sliding_attention" {
			cfg.Sliding = true
			cfg.SlidingWindow = s.Model.Config.SlidingWindow
		}
		layeredIndex[layer] = len(configs)
		configs = append(configs, cfg)
	}
	return dims, configs, layeredIndex, nil
}

func validateLayeredSeedRows(layer, dim, seqLen int, k, v []float32) (int, error) {
	if dim <= 0 {
		if len(k) != 0 || len(v) != 0 {
			return 0, fmt.Errorf("layered KV seed layer %d unexpected non-KV K/V=%d/%d", layer, len(k), len(v))
		}
		return 0, nil
	}
	if len(k) != len(v) {
		return 0, fmt.Errorf("layered KV seed layer %d K/V len=%d/%d", layer, len(k), len(v))
	}
	if len(k)%dim != 0 {
		return 0, fmt.Errorf("layered KV seed layer %d K/V len=%d not divisible by dim=%d", layer, len(k), dim)
	}
	rows := len(k) / dim
	if rows != seqLen {
		return 0, fmt.Errorf("layered KV seed layer %d rows=%d, want output len=%d", layer, rows, seqLen)
	}
	return rows, nil
}

func validateLayeredSeedMaterialization(manager *kv.LayeredF32KV, idx, layer, dim int, sourceK, sourceV []float32) error {
	gotK, gotV, startToken, err := manager.MaterializeLayer(idx)
	if err != nil {
		return fmt.Errorf("layered KV seed layer %d materialize: %w", layer, err)
	}
	if len(gotK) != len(gotV) || len(gotK)%dim != 0 {
		return fmt.Errorf("layered KV seed layer %d materialized K/V=%d/%d incompatible with dim=%d", layer, len(gotK), len(gotV), dim)
	}
	totalRows := len(sourceK) / dim
	gotRows := len(gotK) / dim
	if startToken < 0 || startToken > totalRows || gotRows != totalRows-startToken {
		return fmt.Errorf("layered KV seed layer %d materialized rows start=%d rows=%d, want suffix of %d", layer, startToken, gotRows, totalRows)
	}
	wantOff := startToken * dim
	if !sameFloat32Exact(gotK, sourceK[wantOff:]) || !sameFloat32Exact(gotV, sourceV[wantOff:]) {
		return fmt.Errorf("layered KV seed layer %d materialized rows mismatch", layer)
	}
	return nil
}

func commitAcceptedLayeredKV(cache *kv.LayeredF32KV, cp kv.LayeredF32KVCheckpoint, acceptance MTPAcceptance) error {
	if err := acceptance.Validate(); err != nil {
		return err
	}
	keep := acceptance.KVKeepTokens()
	if keep <= 0 {
		return fmt.Errorf("invalid MTP KV keep token count from accepted prefix %d", acceptance.AcceptedPrefixLen)
	}
	if cache == nil {
		return fmt.Errorf("nil layered KV")
	}
	return cache.KeepAppended(cp, keep)
}

func sameFloat32Exact(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
