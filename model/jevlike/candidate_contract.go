package jevlike

import (
	"fmt"
	"regexp"
	"strings"
)

const SelfContainedCandidatesV1 = "self-contained-v1"
const BenchmarkFragmentsV1 = "benchmark-fragments-v1"

var positionalCandidate = regexp.MustCompile(`(?i)^(?:option |answer |choice )?(?:[a-z]|[0-9]+)[.)]?$`)

// ValidateCandidateContract checks a caller's explicit late-interaction input
// declaration. It cannot establish semantic self-containment: that is a dataset
// author/reviewer obligation, not something tokenisation or regex can prove.
// Independent option encoding cannot resolve a letter, pronoun or fragment by
// looking at the question or other options. Benchmark fragments are opt-in and
// must never be reported as satisfying the self-contained contract.
func ValidateCandidateContract(r DirectChoiceRequest) error {
	if _, err := RenderChoiceContent(r); err != nil {
		return err
	}
	switch r.CandidateContract {
	case BenchmarkFragmentsV1:
		return nil
	case SelfContainedCandidatesV1:
		for _, c := range r.Candidates {
			text := strings.ToLower(strings.TrimSpace(c.Text))
			text = strings.TrimRight(text, ".!?")
			if positionalCandidate.MatchString(text) {
				return fmt.Errorf("candidate %q is a positional code, not self-contained text", c.ID)
			}
			switch text {
			case "yes", "no", "true", "false", "it", "this", "that", "the above", "all of the above", "none of the above", "both", "neither":
				return fmt.Errorf("candidate %q requires question/option context; use an explicit proposition or declared benchmark-fragments-v1", c.ID)
			}
		}
		return nil
	default:
		return fmt.Errorf("declare candidate_contract as self-contained-v1 or explicitly benchmark-fragments-v1")
	}
}
