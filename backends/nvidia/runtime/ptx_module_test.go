package nvidia

import "testing"

func TestOwnedPTXModuleRejectsInvalid(t *testing.T) {
	for _, tc := range []struct {
		ptx   string
		names []string
	}{{"", []string{"a"}}, {"x", nil}, {"x", []string{""}}, {"x", []string{"a", "a"}}} {
		if m, e := LoadPTXFunctions(tc.ptx, tc.names); e == nil || m != nil {
			t.Fatal("invalid module accepted")
		}
	}
	var m *PTXModule
	if m.Function("a") != 0 {
		t.Fatal("nil handle")
	}
	if e := m.Close(); e != nil {
		t.Fatal(e)
	}
	m = &PTXModule{}
	if m.Function("a") != 0 {
		t.Fatal("empty handle")
	}
	if e := m.Close(); e != nil {
		t.Fatal(e)
	}
}
