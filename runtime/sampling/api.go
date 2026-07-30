// Package sampling implements bounded top-k / top-p / temperature sampling
// over autoregressive next-token logits.
//
// The package is deliberately small and generic: it knows nothing about
// tokenizers, KV caches, or model call sites. Callers pass a []float32 of
// per-token logits (indexed by token ID) plus a Config, and get back the
// chosen token ID.
//
// # Determinism contract
//
// SampleWithDraw is the deterministic primitive: for a given (logits, cfg,
// unitDraw) triple the result is 100% reproducible, independent of map
// iteration order, goroutine scheduling, or Go version, because:
//
//   - Candidates are always ordered by descending logit, then ascending
//     token ID, before any threshold or cumulative-probability decision is
//     made.
//   - unitDraw is a single float64 in [0, 1] representing the caller's random
//     draw. It is clamped into range before use.
//
// Sample is a convenience wrapper that pulls exactly one float64 out of a
// caller-provided *rand.Rand (via Float64) and forwards it to
// SampleWithDraw. Using a *rand.Rand seeded deterministically (e.g.
// rand.New(rand.NewSource(seed))) makes Sample fully reproducible across
// runs; tests that need to avoid any *rand.Rand entirely should call
// SampleWithDraw directly with a fixed unitDraw.
package sampling

import (
	"errors"
	"fmt"
	"math"
)

// Config controls autoregressive next-token sampling behavior.
//
// Contract:
//
//   - Temperature <= 0 selects greedy decoding: the highest-logit valid
//     token wins, ties broken by ascending token ID. TopK and TopP are
//     ignored in this mode.
//   - TopK == 0 means unlimited: no top-k truncation is applied. TopK > 0
//     bounds the candidate set to the K highest-logit valid tokens (ties at
//     the K-th boundary are broken toward ascending token ID, i.e. lower
//     token IDs are kept).
//   - TopP == 0 or TopP == 1 means top-p (nucleus) truncation is disabled.
//     TopP in (0, 1) truncates the descending-probability candidate list to
//     the smallest prefix whose cumulative probability is >= TopP, always
//     including the token that crosses the threshold.
//   - When both are enabled, TopK is applied first, then TopP is applied to
//     the surviving candidates (standard top-k-then-top-p composition).
//   - NaN logits are always excluded from consideration. -Inf logits are
//     excluded (they carry zero probability by definition). +Inf logits are
//     handled deterministically: all tokens tied at +Inf share equal
//     probability mass and every finite token receives zero mass.
type Config struct {
	// Temperature scales logits before softmax. <= 0 means greedy decoding.
	Temperature float64
	// TopK bounds the candidate set to the K highest-logit tokens. 0 means
	// unlimited.
	TopK int
	// TopP bounds the candidate set via cumulative probability (nucleus
	// sampling). 0 or 1 means disabled.
	TopP float64
}

// Result is the outcome of a single sampling decision.
type Result struct {
	// TokenID is the chosen token's index into the input logits slice.
	TokenID int
	// Logit is the original (untransformed) logit value of the chosen
	// token.
	Logit float32
	// Candidates is the number of tokens that were eligible for the final
	// selection step: for greedy decoding this is the count of valid
	// (non-NaN, non -Inf) logits; for temperature sampling this is the size
	// of the candidate set after TopK/TopP truncation.
	Candidates int
}

// Sentinel errors returned by Sample / SampleWithDraw. Config validation
// errors wrap ErrInvalidConfig, so errors.Is(err, ErrInvalidConfig) reports
// true for any config problem.
var (
	// ErrEmptyLogits is returned when the logits slice has zero length.
	ErrEmptyLogits = errors.New("sampling: logits slice is empty")
	// ErrAllInvalid is returned when every logit is NaN or -Inf, leaving no
	// valid token to select.
	ErrAllInvalid = errors.New("sampling: all logits are NaN or -Inf")
	// ErrInvalidConfig is the wrapped base error for malformed Config
	// values (negative TopK, out-of-range TopP, NaN Temperature).
	ErrInvalidConfig = errors.New("sampling: invalid config")
	// ErrNilRand is returned by Sample when Temperature > 0 (stochastic
	// sampling is requested) but no *rand.Rand was provided.
	ErrNilRand = errors.New("sampling: rand is required when temperature > 0")
)

// Validate reports whether c is well-formed. It does not consider
// Temperature <= 0 an error (that is the documented greedy mode).
func (c Config) Validate() error {
	if c.TopK < 0 {
		return fmt.Errorf("%w: TopK must be >= 0, got %d", ErrInvalidConfig, c.TopK)
	}
	if math.IsNaN(c.TopP) || c.TopP < 0 || c.TopP > 1 {
		return fmt.Errorf("%w: TopP must be in [0, 1], got %v", ErrInvalidConfig, c.TopP)
	}
	if math.IsNaN(c.Temperature) {
		return fmt.Errorf("%w: Temperature must not be NaN", ErrInvalidConfig)
	}
	return nil
}

// topPEnabled reports whether nucleus truncation is active for c.
func (c Config) topPEnabled() bool {
	return c.TopP > 0 && c.TopP < 1
}
