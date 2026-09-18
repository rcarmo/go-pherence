package jevlike

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"testing"
)

func TestCheckpointRoundTrip(t *testing.T) {
	m, err := NewTinyScorer(Config{Width: 4, Rank: 3, ContextTokens: 8, OptionTokens: 4})
	if err != nil {
		t.Fatal(err)
	}
	if err = InitializeTinyScorer(m, 7); err != nil {
		t.Fatal(err)
	}
	c, err := TinyCheckpoint(m)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "nested", "model.json")
	if err = SaveCheckpoint(path, c); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}
	other, err := loaded.Tiny()
	if err != nil {
		t.Fatal(err)
	}
	b, err := BuildByteBatch([]ChoiceExample{{Context: "abc", Options: []string{"a", "b"}, Label: 0}}, 8, 4)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := m.Forward(b)
	got, _ := other.Forward(b)
	for i, v := range want[0] {
		if got[0][i] != v {
			t.Fatalf("roundtrip logit mismatch %g %g", v, got[0][i])
		}
	}
}
func TestCheckpointRejectsMalformed(t *testing.T) {
	m, _ := NewTinyScorer(Config{Width: 2, Rank: 2, ContextTokens: 4, OptionTokens: 2})
	c, _ := TinyCheckpoint(m)
	c.Parameters = c.Parameters[:len(c.Parameters)-1]
	if c.Validate() == nil {
		t.Fatal("accepted missing tensor")
	}
	c, _ = TinyCheckpoint(m)
	c.Parameters[1] = c.Parameters[0]
	if c.Validate() == nil {
		t.Fatal("accepted duplicate tensor")
	}
	c, _ = TinyCheckpoint(m)
	c.Parameters[0].Shape[0]++
	if c.Validate() == nil {
		t.Fatal("accepted wrong shape")
	}
	c, _ = TinyCheckpoint(m)
	raw, _ := json.Marshal(c)
	if _, err := ReadCheckpoint(bytes.NewReader(append(raw, []byte(" {}")...))); err == nil {
		t.Fatal("accepted trailing JSON")
	}
	c.Version = 2
	if c.Validate() == nil {
		t.Fatal("accepted unknown version")
	}
}
