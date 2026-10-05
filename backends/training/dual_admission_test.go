package training

import (
	"testing"
)

func TestTrainingDualAdmissionBranches(t *testing.T) {
	if err := DualModShape(1, 0, 0); err != nil {
		t.Fatal(err)
	}
	if err := DualNormForwardShape(1, 2, 0, 0, 1e-4); err != nil {
		t.Fatal(err)
	}
	if DualNormForwardShape(0, 2, 0, 0, 1e-4) == nil {
		t.Fatal("invalid forward shape")
	}
}
