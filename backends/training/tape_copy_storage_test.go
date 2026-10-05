package training

import "testing"

func TestTrainingLeaseMetadataDropsOnlyValues(t *testing.T) {
	source, _, _ := handoffFixture()
	copy := copyLeaseMetadata(source)
	for _, s := range copy.Tensors {
		if s.Values != nil {
			t.Fatal("retained host constant")
		}
	}
	source.Tensors[0].Shape[0] = 9
	source.Commands[0].Inputs[0] = 1
	source.Outputs[0] = 0
	if copy.Tensors[0].Shape[0] != 2 || copy.Commands[0].Inputs[0] != 0 || copy.Outputs[0] != 2 {
		t.Fatal("provenance borrowed")
	}
}
func TestTrainingExternalCopyOwnsRemainingValues(t *testing.T) {
	source, _, _ := handoffFixture()
	out, e := WithExternalTapeInputs(source, []int{0}, 1024)
	if e != nil {
		t.Fatal(e)
	}
	source.Tensors[1].Values[0] = 99
	if out.Tensors[0].Values != nil || !out.Tensors[0].External || out.Tensors[1].Values[0] != 3 {
		t.Fatal("external conversion ownership")
	}
}
