package jevlike

import "testing"

type fakeEncoder struct{}

func (fakeEncoder) Encode(text string, n int) ([][]float32, error) {
	var rows [][]float32
	for i, b := range []byte(text) {
		if i >= n {
			break
		}
		rows = append(rows, []float32{float32(b), float32(i + 1), -float32(b), 1})
	}
	if len(rows) == 0 {
		rows = [][]float32{{1, 2, 3, 4}}
	}
	return rows, nil
}
func TestFrozenPoolingAndShuffle(t *testing.T) {
	tiny, _ := NewInitializedTinyScorer(Config{Width: 4, Rank: 3, ContextTokens: 8, OptionTokens: 4}, 7)
	c := Checkpoint{Version: 1, Encoder: "frozen", EncoderReference: "test", Config: tiny.Config, Parameters: tiny.Head.NamedParameters("head")}
	m, err := c.Frozen(fakeEncoder{})
	if err != nil {
		t.Fatal(err)
	}
	ex := []ChoiceExample{{Context: "abc", Options: []string{"ab", "c"}, Label: 0}, {Context: "d", Options: []string{"a", "bc", "d"}, Label: 1}}
	a, err := m.Forward(ex, true)
	if err != nil {
		t.Fatal(err)
	}
	ex[0].Context, ex[1].Context = ex[1].Context, ex[0].Context
	b, err := m.Forward(ex, false)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a {
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				t.Fatal("shuffle mismatch")
			}
		}
	}
	if a[0][2] != maskedFillValue {
		t.Fatal("option padding not masked")
	}
}
