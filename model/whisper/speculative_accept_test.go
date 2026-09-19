package whisper

import "testing"

func TestAcceptDraftTokens(t *testing.T) {
	got, err := AcceptDraftTokens([]int{1, 2, 3}, []int{1, 2, 9, 10})
	if err != nil {
		t.Fatal(err)
	}
	if got.AllAccepted || got.RejectedAt != 2 || got.Bonus != 9 || len(got.Accepted) != 2 {
		t.Fatalf("acceptance=%+v", got)
	}
	got, err = AcceptDraftTokens([]int{1, 2}, []int{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	if !got.AllAccepted || got.RejectedAt != -1 || got.Bonus != 3 || len(got.Accepted) != 2 {
		t.Fatalf("all acceptance=%+v", got)
	}
	if _, err := AcceptDraftTokens([]int{1}, []int{1}); err == nil {
		t.Fatal("accepted malformed verifier length")
	}
}
