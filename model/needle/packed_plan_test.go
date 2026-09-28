package needle

import (
	"reflect"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/backends/simd/runtime"
	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

func TestPackedOnlyPlanTinyArchive(t *testing.T) {
	m, _, err := LoadArchive("../../loader/needle/testdata/needle3.cact")
	if err != nil {
		t.Fatal(err)
	}
	before := m.Checkpoint()
	plan, err := planPackedOnly(m)
	if err != nil {
		t.Fatal(err)
	}
	again, err := planPackedOnly(m)
	if err != nil || !reflect.DeepEqual(plan, again) || !reflect.DeepEqual(before, m.Checkpoint()) {
		t.Fatal("plan changed model or depended on map order", err)
	}
	var covered, dense int64
	for _, entry := range plan.PotentiallyReplaceable {
		covered += entry.Bytes
		if entry.Why == "" {
			t.Fatal("unexplained candidate")
		}
	}
	for _, entry := range plan.RequiredDecoded {
		dense += entry.Bytes
		if entry.Why == "" {
			t.Fatal("unexplained dense tensor")
		}
	}
	if covered != 14592 || dense != 10420 || len(plan.PotentiallyReplaceable)+len(plan.RequiredDecoded) != 69 {
		t.Fatalf("coverage candidates=%d dense=%d names=%d", covered, dense, len(plan.PotentiallyReplaceable)+len(plan.RequiredDecoded))
	}
	if len(plan.Blockers) != 5 {
		t.Fatalf("missing runtime blockers: %v", plan.Blockers)
	}
	for _, name := range []string{"embedding/embedding", "stack/layers/block/self_attn/q_proj/kernel"} {
		if !containsPlanTensor(plan.PotentiallyReplaceable, name) {
			t.Fatalf("missing packed candidate %s", name)
		}
	}
	for _, name := range []string{"stack/final_norm/scale", "stack/layers/block/hadamard_mlp/d1", "embedding_head/proj/kernel", "confidence_head/proj/kernel"} {
		if !containsPlanTensor(plan.RequiredDecoded, name) {
			t.Fatalf("missing dense requirement %s", name)
		}
	}
}

func containsPlanTensor(entries []packedPlanTensor, name string) bool {
	for _, entry := range entries {
		if entry.Name == name {
			return true
		}
	}
	return false
}

func TestPackedOnlyPlanPartialAndMalformed(t *testing.T) {
	m, _, err := LoadArchive("../../loader/needle/testdata/needle3.cact")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = planPackedOnly(nil); err == nil {
		t.Fatal("accepted nil")
	}
	copyModel := *m
	copyModel.deployed = false
	if _, err = planPackedOnly(&copyModel); err == nil {
		t.Fatal("accepted undeployed")
	}
	var chosen packedKey
	for key := range m.packed {
		if key.layer >= 0 {
			chosen = key
			break
		}
	}
	if chosen.name == "" {
		t.Fatal("no layered packed matrix")
	}
	copyModel = *m
	copyModel.packed = make(map[packedKey]*simd.CQMatrix, len(m.packed))
	for key, p := range m.packed {
		copyModel.packed[key] = p
	}
	delete(copyModel.packed, chosen)
	plan, err := planPackedOnly(&copyModel)
	if err != nil {
		t.Fatal(err)
	}
	if containsPlanTensor(plan.PotentiallyReplaceable, chosen.name) {
		t.Fatal("partial layered tensor marked replaceable")
	}
	found := false
	for _, entry := range plan.RequiredDecoded {
		if entry.Name == chosen.name && strings.Contains(entry.Why, "partial tensors stay decoded") {
			found = true
		}
	}
	if !found {
		t.Fatal("missing explicit partial-coverage reason")
	}
	if _, err = planPackedOnly(m); err != nil {
		t.Fatal("mutated original model")
	}
	copyModel.packed[packedKey{"not-a-tensor", -1}] = m.packed[chosen]
	if _, err = planPackedOnly(&copyModel); err == nil {
		t.Fatal("accepted orphan packed key")
	}
	delete(copyModel.packed, packedKey{"not-a-tensor", -1})
	copyModel.packed[packedKey{"stack/final_norm/scale", -1}] = m.packed[chosen]
	if _, err = planPackedOnly(&copyModel); err == nil {
		t.Fatal("accepted unused packed key")
	}
	delete(copyModel.packed, packedKey{"stack/final_norm/scale", -1})
	copyModel.packed[packedKey{chosen.name, -1}] = m.packed[chosen]
	if _, err = planPackedOnly(&copyModel); err == nil {
		t.Fatal("accepted invalid layered key")
	}
	copyModel = *m
	copyModel.tensors = make(map[string]checkpoint.Tensor, len(m.tensors))
	for name, tensor := range m.tensors {
		copyModel.tensors[name] = tensor
	}
	delete(copyModel.tensors, "stack/final_norm/scale")
	if _, err = planPackedOnly(&copyModel); err == nil {
		t.Fatal("accepted missing expected tensor")
	}
	copyModel.tensors["stack/final_norm/scale"] = checkpoint.Tensor{Shape: []int{1, 2}}
	if _, err = planPackedOnly(&copyModel); err == nil {
		t.Fatal("accepted malformed expected shape")
	}
}
