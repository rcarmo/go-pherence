package model

import (
	"fmt"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

// MTPVerifierTokens returns the token sequence the main verifier must process:
// the previous/input token followed by G drafter candidates.
func MTPVerifierTokens(inputToken int, drafted []int) ([]int, error) {
	if inputToken < 0 {
		return nil, fmt.Errorf("input token %d out of range", inputToken)
	}
	for i, tok := range drafted {
		if tok < 0 {
			return nil, fmt.Errorf("drafted token %d at index %d out of range", tok, i)
		}
	}
	tokens := make([]int, 0, len(drafted)+1)
	tokens = append(tokens, inputToken)
	tokens = append(tokens, drafted...)
	return tokens, nil
}

// MTPVerifierResult is the contract filled by a main-model verifier forward.
// Logits must contain one row per verifier position: G+1 rows for G drafted
// tokens. ActivationRows, when present, contains the final main-model activation
// for each verifier row. FinalActivation is kept as the final verifier-row
// activation for compatibility; MTP graph generation seeds the next drafter
// from the committed activation row selected by acceptance.
type MTPVerifierResult struct {
	InputToken      int
	DraftedTokens   []int
	VerifierTokens  []int // [inputToken] + draftedTokens
	Logits          [][]float32
	ActivationRows  [][]float32
	FinalActivation []float32
	Acceptance      MTPAcceptance
}

// NewMTPVerifierResult validates verifier outputs, derives greedy acceptance,
// and copies slice headers/data that callers commonly mutate after verification.
func NewMTPVerifierResult(inputToken int, drafted []int, logits [][]float32, finalActivation []float32) (MTPVerifierResult, error) {
	return newMTPVerifierResult(inputToken, drafted, logits, finalActivation, nil, 0, 0)
}

// NewMTPVerifierResultForModel validates verifier outputs against model-owned
// dimensions. It is intended for the real verifier path; tests and low-level
// helpers may keep using NewMTPVerifierResult when no model is available.
func NewMTPVerifierResultForModel(m *LlamaModel, inputToken int, drafted []int, logits [][]float32, finalActivation []float32) (MTPVerifierResult, error) {
	vocab, hidden, err := mtpVerifierModelDims(m)
	if err != nil {
		return MTPVerifierResult{}, err
	}
	return newMTPVerifierResult(inputToken, drafted, logits, finalActivation, nil, vocab, hidden)
}

// NewMTPVerifierResultRowsForModel validates a full verifier batch result with
// one final activation row per verifier position. This is the graph-generation
// path: acceptance selects which activation row seeds the next drafter cycle.
func NewMTPVerifierResultRowsForModel(m *LlamaModel, inputToken int, drafted []int, logits [][]float32, activationRows [][]float32) (MTPVerifierResult, error) {
	vocab, hidden, err := mtpVerifierModelDims(m)
	if err != nil {
		return MTPVerifierResult{}, err
	}
	return newMTPVerifierResult(inputToken, drafted, logits, nil, activationRows, vocab, hidden)
}

func mtpVerifierModelDims(m *LlamaModel) (vocab, hidden int, err error) {
	if m == nil {
		return 0, 0, fmt.Errorf("nil model")
	}
	vocab, hidden = m.Config.VocabSize, m.Config.HiddenSize
	if vocab <= 0 || hidden <= 0 {
		return 0, 0, fmt.Errorf("invalid verifier model dims vocab=%d hidden=%d", vocab, hidden)
	}
	return vocab, hidden, nil
}

func newMTPVerifierResult(inputToken int, drafted []int, logits [][]float32, finalActivation []float32, activationRows [][]float32, vocab, hidden int) (MTPVerifierResult, error) {
	verifierTokens, err := MTPVerifierTokens(inputToken, drafted)
	if err != nil {
		return MTPVerifierResult{}, err
	}
	if vocab > 0 {
		for i, tok := range verifierTokens {
			if tok >= vocab {
				return MTPVerifierResult{}, fmt.Errorf("verifier token %d at index %d out of range [0,%d)", tok, i, vocab)
			}
		}
	}
	if len(logits) != len(drafted)+1 {
		return MTPVerifierResult{}, fmt.Errorf("verifier logits rows=%d, want drafted+1=%d", len(logits), len(drafted)+1)
	}
	copiedActivationRows := make([][]float32, 0, len(activationRows))
	if len(activationRows) > 0 {
		if len(activationRows) != len(logits) {
			return MTPVerifierResult{}, fmt.Errorf("verifier activation rows=%d, want logits rows=%d", len(activationRows), len(logits))
		}
		for i, row := range activationRows {
			if hidden > 0 && len(row) != hidden {
				return MTPVerifierResult{}, fmt.Errorf("verifier activation row %d len=%d, want %d", i, len(row), hidden)
			}
			copiedActivationRows = append(copiedActivationRows, append([]float32(nil), row...))
		}
		finalActivation = copiedActivationRows[len(copiedActivationRows)-1]
	} else if hidden > 0 && len(finalActivation) != hidden {
		return MTPVerifierResult{}, fmt.Errorf("final activation len=%d, want %d", len(finalActivation), hidden)
	}
	copiedLogits := make([][]float32, len(logits))
	for i, row := range logits {
		if len(row) == 0 {
			return MTPVerifierResult{}, fmt.Errorf("verifier logits row %d is empty", i)
		}
		if vocab > 0 && len(row) != vocab {
			return MTPVerifierResult{}, fmt.Errorf("verifier logits row %d len=%d, want vocab=%d", i, len(row), vocab)
		}
		copiedLogits[i] = append([]float32(nil), row...)
	}
	acceptance, err := AcceptMTPDraftFromLogits(drafted, copiedLogits)
	if err != nil {
		return MTPVerifierResult{}, err
	}
	return MTPVerifierResult{
		InputToken:      inputToken,
		DraftedTokens:   append([]int(nil), drafted...),
		VerifierTokens:  verifierTokens,
		Logits:          copiedLogits,
		ActivationRows:  copiedActivationRows,
		FinalActivation: append([]float32(nil), finalActivation...),
		Acceptance:      acceptance,
	}, nil
}

// CommittedActivation returns the verifier activation row corresponding to the
// accepted-prefix+bonus commit. The row index is AcceptedPrefixLen because that
// verifier row's logits emitted the bonus/next output token. Legacy verifier
// results without ActivationRows can only describe the all-drafts-accepted row.
func (r MTPVerifierResult) CommittedActivation() ([]float32, error) {
	if err := r.Acceptance.Validate(); err != nil {
		return nil, err
	}
	idx := r.Acceptance.AcceptedPrefixLen
	if len(r.ActivationRows) > 0 {
		if idx >= len(r.ActivationRows) {
			return nil, fmt.Errorf("committed activation row=%d outside rows=%d", idx, len(r.ActivationRows))
		}
		return r.ActivationRows[idx], nil
	}
	if idx != len(r.DraftedTokens) {
		return nil, fmt.Errorf("missing committed activation row for accepted prefix=%d drafted=%d", idx, len(r.DraftedTokens))
	}
	if len(r.FinalActivation) == 0 {
		return nil, fmt.Errorf("missing verifier final activation")
	}
	return r.FinalActivation, nil
}

// CommitFloatKV applies the verifier result's acceptance to staged uncompressed
// KV caches. The checkpoint must be from immediately before the verifier pass.
func (r MTPVerifierResult) CommitFloatKV(m *LlamaModel, kvCacheK, kvCacheV [][]float32, cp kv.FloatKVCheckpoint) error {
	if m == nil {
		return fmt.Errorf("nil model")
	}
	if err := r.validateAcceptanceMatchesLogits(); err != nil {
		return err
	}
	return m.CommitAcceptedFloatKV(kvCacheK, kvCacheV, cp, r.Acceptance)
}

// CommitGraphFloatKV applies acceptance through the explicit MTPExecutionGraph,
// then commits exactly the graph's accepted-prefix+bonus KV window.
func (r MTPVerifierResult) CommitGraphFloatKV(m *LlamaModel, graph MTPExecutionGraph, kvCacheK, kvCacheV [][]float32, cp kv.FloatKVCheckpoint) (MTPKVCommitPlan, error) {
	if m == nil {
		return MTPKVCommitPlan{}, fmt.Errorf("nil model")
	}
	if err := r.validateGraphForModel(m, graph); err != nil {
		return MTPKVCommitPlan{}, err
	}
	commit, err := graph.CommitPlan(r.Acceptance)
	if err != nil {
		return MTPKVCommitPlan{}, err
	}
	if err := m.CommitAcceptedFloatKV(kvCacheK, kvCacheV, cp, r.Acceptance); err != nil {
		return MTPKVCommitPlan{}, err
	}
	return commit, nil
}

// CommitCompressedKV applies the verifier result's acceptance to staged
// compressed/TurboQuant KV caches. The checkpoints must be from immediately
// before the verifier pass.
func (r MTPVerifierResult) CommitCompressedKV(caches []*kv.CompressedKVCache, cp []kv.CompressedKVCheckpoint) error {
	if err := r.validateAcceptanceMatchesLogits(); err != nil {
		return err
	}
	return CommitAcceptedCompressedKV(caches, cp, r.Acceptance)
}

// CommitGraphCompressedKV is the compressed/TurboQuant counterpart to
// CommitGraphFloatKV. It validates the explicit graph before retaining the
// accepted-prefix+bonus compressed KV window. Callers that own a verifier model
// should prefer CommitGraphCompressedKVForModel so token/logit/activation widths
// are checked against model metadata before KV mutation.
func (r MTPVerifierResult) CommitGraphCompressedKV(graph MTPExecutionGraph, caches []*kv.CompressedKVCache, cp []kv.CompressedKVCheckpoint) (MTPKVCommitPlan, error) {
	if err := r.validateGraph(graph); err != nil {
		return MTPKVCommitPlan{}, err
	}
	return r.commitGraphCompressedKVUncheckedModel(graph, caches, cp)
}

// CommitGraphCompressedKVForModel applies compressed/TurboQuant graph commits
// with the same model-aware verifier metadata checks as CommitGraphFloatKV.
func (r MTPVerifierResult) CommitGraphCompressedKVForModel(m *LlamaModel, graph MTPExecutionGraph, caches []*kv.CompressedKVCache, cp []kv.CompressedKVCheckpoint) (MTPKVCommitPlan, error) {
	if m == nil {
		return MTPKVCommitPlan{}, fmt.Errorf("nil model")
	}
	if err := r.validateGraphForModel(m, graph); err != nil {
		return MTPKVCommitPlan{}, err
	}
	return r.commitGraphCompressedKVUncheckedModel(graph, caches, cp)
}

func (r MTPVerifierResult) commitGraphCompressedKVUncheckedModel(graph MTPExecutionGraph, caches []*kv.CompressedKVCache, cp []kv.CompressedKVCheckpoint) (MTPKVCommitPlan, error) {
	commit, err := graph.CommitPlan(r.Acceptance)
	if err != nil {
		return MTPKVCommitPlan{}, err
	}
	if err := CommitAcceptedCompressedKV(caches, cp, r.Acceptance); err != nil {
		return MTPKVCommitPlan{}, err
	}
	return commit, nil
}

func (r MTPVerifierResult) validateGraphForModel(m *LlamaModel, graph MTPExecutionGraph) error {
	if m == nil {
		return fmt.Errorf("nil model")
	}
	if err := r.validateGraph(graph); err != nil {
		return err
	}
	vocab, hidden := m.Config.VocabSize, m.Config.HiddenSize
	if vocab <= 0 || hidden <= 0 {
		return fmt.Errorf("invalid MTP verifier model dims vocab=%d hidden=%d", vocab, hidden)
	}
	for i, tok := range r.VerifierTokens {
		if tok >= vocab {
			return fmt.Errorf("MTP verifier token %d at index %d out of range [0,%d)", tok, i, vocab)
		}
	}
	for i, row := range r.Logits {
		if len(row) != vocab {
			return fmt.Errorf("MTP verifier logits row %d len=%d, want vocab=%d", i, len(row), vocab)
		}
	}
	for i, row := range r.ActivationRows {
		if len(row) != hidden {
			return fmt.Errorf("MTP verifier activation row %d len=%d, want hidden=%d", i, len(row), hidden)
		}
	}
	if len(r.FinalActivation) > 0 && len(r.FinalActivation) != hidden {
		return fmt.Errorf("MTP verifier final activation len=%d, want hidden=%d", len(r.FinalActivation), hidden)
	}
	return nil
}

func (r MTPVerifierResult) validateGraph(graph MTPExecutionGraph) error {
	if r.InputToken != graph.InputToken || graph.Verifier.InputToken != r.InputToken {
		return fmt.Errorf("MTP graph input token mismatch result=%d graph=%d verifier=%d", r.InputToken, graph.InputToken, graph.Verifier.InputToken)
	}
	if !mtpSameInts(r.DraftedTokens, graph.DraftedTokens) || !mtpSameInts(r.DraftedTokens, graph.Verifier.DraftedTokens) {
		return fmt.Errorf("MTP graph drafted tokens mismatch result=%v graph=%v verifier=%v", r.DraftedTokens, graph.DraftedTokens, graph.Verifier.DraftedTokens)
	}
	if !mtpSameInts(r.VerifierTokens, graph.Verifier.VerifierTokens) {
		return fmt.Errorf("MTP graph verifier tokens mismatch result=%v graph=%v", r.VerifierTokens, graph.Verifier.VerifierTokens)
	}
	if len(graph.Verifier.Positions) != len(r.VerifierTokens) {
		return fmt.Errorf("MTP graph verifier positions=%d, verifier tokens=%d", len(graph.Verifier.Positions), len(r.VerifierTokens))
	}
	if len(r.Logits) != len(r.VerifierTokens) {
		return fmt.Errorf("MTP verifier logits rows=%d, verifier tokens=%d", len(r.Logits), len(r.VerifierTokens))
	}
	if len(r.ActivationRows) > 0 && len(r.ActivationRows) != len(r.VerifierTokens) {
		return fmt.Errorf("MTP verifier activation rows=%d, verifier tokens=%d", len(r.ActivationRows), len(r.VerifierTokens))
	}
	if err := r.validateAcceptanceMatchesLogits(); err != nil {
		return err
	}
	return nil
}

func (r MTPVerifierResult) validateAcceptanceMatchesLogits() error {
	want, err := AcceptMTPDraftFromLogits(r.DraftedTokens, r.Logits)
	if err != nil {
		return err
	}
	if !sameMTPAcceptance(r.Acceptance, want) {
		return fmt.Errorf("MTP verifier acceptance %+v does not match logits-derived acceptance %+v", r.Acceptance, want)
	}
	return nil
}

func sameMTPAcceptance(a, b MTPAcceptance) bool {
	return a.DraftedCount == b.DraftedCount &&
		a.VerifiedCount == b.VerifiedCount &&
		a.AcceptedPrefixLen == b.AcceptedPrefixLen &&
		a.BonusToken == b.BonusToken &&
		a.AllDraftsAccepted == b.AllDraftsAccepted &&
		a.FirstRejectedIndex == b.FirstRejectedIndex &&
		mtpSameInts(a.AcceptedTokens, b.AcceptedTokens) &&
		mtpSameInts(a.OutputTokens, b.OutputTokens)
}

func mtpSameInts(a, b []int) bool {
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
