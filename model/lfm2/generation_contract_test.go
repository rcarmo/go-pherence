package lfm2

import "testing"

func testGenerationContractPlan(t *testing.T) RuntimeRequestPlan {
	t.Helper()
	_, plan := testGenerationContractPlanConfig(t)
	return plan
}

func TestGenerationExecutionContract(t *testing.T) {
	plan := testGenerationContractPlan(t)
	contract, err := NewGenerationExecutionContract(plan)
	if err != nil {
		t.Fatal(err)
	}
	if contract.PromptTokens != 3 || contract.MaxNewTokens != 4 || contract.MaxSequence != 7 {
		t.Fatalf("contract=%+v", contract)
	}
	if err := contract.ValidatePrompt([]uint32{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	if err := contract.ValidateOutput([]uint32{4, 5}); err != nil {
		t.Fatal(err)
	}
}

func TestGenerationExecutionContractRejectsMalformed(t *testing.T) {
	plan := testGenerationContractPlan(t)
	contract, err := NewGenerationExecutionContract(plan)
	if err != nil {
		t.Fatal(err)
	}
	if err := contract.ValidatePrompt([]uint32{1, 2}); err == nil {
		t.Fatal("expected prompt length error")
	}
	if err := contract.ValidatePrompt([]uint32{1, 2, uint32(contract.Context.VocabSize)}); err == nil {
		t.Fatal("expected prompt vocab error")
	}
	if err := contract.ValidateOutput(nil); err == nil {
		t.Fatal("expected empty output error")
	}
	if err := contract.ValidateOutput([]uint32{1, 2, 3, 4, 5}); err == nil {
		t.Fatal("expected output max error")
	}
	if err := contract.ValidateOutput([]uint32{uint32(contract.Context.VocabSize)}); err == nil {
		t.Fatal("expected output vocab error")
	}
}
