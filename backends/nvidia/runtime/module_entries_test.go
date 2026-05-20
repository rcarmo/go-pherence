package nvidia

import "testing"

func TestValidateModuleEntriesRejectsMalformedInputs(t *testing.T) {
	if err := validateModuleEntries(nil); err == nil {
		t.Fatal("accepted empty entry table")
	}
	if err := validateModuleEntries([]moduleEntry{{name: "", ptx: "// ptx"}}); err == nil {
		t.Fatal("accepted empty kernel name")
	}
	if err := validateModuleEntries([]moduleEntry{{name: "kernel", ptx: "   "}}); err == nil {
		t.Fatal("accepted empty PTX")
	}
	if err := validateModuleEntries([]moduleEntry{{name: "kernel", ptx: "// a"}, {name: "kernel", ptx: "// b"}}); err == nil {
		t.Fatal("accepted duplicate kernel name")
	}
}

func TestMegaModuleEntriesValidate(t *testing.T) {
	if err := validateModuleEntries(megaModuleEntries()); err != nil {
		t.Fatalf("megaModuleEntries invalid: %v", err)
	}
}
