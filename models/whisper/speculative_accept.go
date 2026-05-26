package whisper

import "fmt"

// DraftAcceptance is the deterministic greedy accept/reject result for a
// speculative verifier pass. Verifier IDs must have len(drafted)+1 entries:
// verifier[0:len(drafted)] verify the drafted tokens, and verifier[len(drafted)]
// is the bonus token when all drafts are accepted.
type DraftAcceptance struct {
	Accepted    []int
	Bonus       int
	AllAccepted bool
	RejectedAt  int // -1 when all drafts are accepted
}

func AcceptDraftTokens(drafted, verifier []int) (DraftAcceptance, error) {
	if len(verifier) != len(drafted)+1 {
		return DraftAcceptance{}, fmt.Errorf("verifier token count=%d want drafted+1=%d", len(verifier), len(drafted)+1)
	}
	for i, tok := range drafted {
		if tok < 0 {
			return DraftAcceptance{}, fmt.Errorf("draft token %d at %d out of range", tok, i)
		}
	}
	for i, tok := range verifier {
		if tok < 0 {
			return DraftAcceptance{}, fmt.Errorf("verifier token %d at %d out of range", tok, i)
		}
	}
	accepted := 0
	for accepted < len(drafted) && drafted[accepted] == verifier[accepted] {
		accepted++
	}
	bonus := verifier[accepted]
	all := accepted == len(drafted)
	rejectedAt := accepted
	if all {
		rejectedAt = -1
	}
	return DraftAcceptance{Accepted: append([]int(nil), drafted[:accepted]...), Bonus: bonus, AllAccepted: all, RejectedAt: rejectedAt}, nil
}
