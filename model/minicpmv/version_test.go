package minicpmv

import "testing"

func TestSupportVersion(t *testing.T) {
	if SupportVersion == "" || RuntimeStatusPending == "" {
		t.Fatalf("empty support version/status")
	}
}
