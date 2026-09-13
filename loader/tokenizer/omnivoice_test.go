package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

type omniVoiceFixture struct {
	Cases []struct {
		Text string `json:"text"`
		IDs  []int  `json:"ids"`
	} `json:"cases"`
}

func loadOmniVoiceFixture(t *testing.T) omniVoiceFixture {
	t.Helper()
	raw, err := os.ReadFile(filepath.Clean("../../testdata/omnivoice/tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	var f omniVoiceFixture
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if len(f.Cases) < 10 {
		t.Fatalf("fixture too small: %d cases", len(f.Cases))
	}
	return f
}

func checkOmniVoiceCases(t *testing.T, tok *Tokenizer, cases []struct {
	Text string `json:"text"`
	IDs  []int  `json:"ids"`
}) {
	t.Helper()
	for _, c := range cases {
		if got := tok.Encode(c.Text); !reflect.DeepEqual(got, c.IDs) {
			t.Errorf("%q\ngot  %v\nwant %v", c.Text, got, c.IDs)
		}
	}
}

func TestOmniVoiceTokenizerFixtureParity(t *testing.T) {
	fixture := loadOmniVoiceFixture(t)
	tok, err := Load("../../testdata/omnivoice/tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	checkOmniVoiceCases(t, tok, fixture.Cases)
}

func TestOmniVoiceRealTokenizerParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_REAL_OMNIVOICE")
	if path == "" {
		t.Skip("set GO_PHERENCE_REAL_OMNIVOICE to a model dir")
	}
	fixture := loadOmniVoiceFixture(t)
	tok, err := Load(filepath.Join(path, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	checkOmniVoiceCases(t, tok, fixture.Cases)
}

func TestSingleDigitPreTokenization(t *testing.T) {
	tok := &Tokenizer{Vocab: map[string]int{"1": 1, "2": 2, "12": 12}, Merges: [][2]string{{"1", "2"}}, AddedSpecial: map[string]int{}, byteLevelMode: byteLevelQwenSingleDigits}
	if got := tok.Encode("12"); !reflect.DeepEqual(got, []int{1, 2}) {
		t.Fatal(got)
	}
	for i := 0; i < 3; i++ {
		if got := tok.Encode("12"); !reflect.DeepEqual(got, []int{1, 2}) {
			t.Fatal(got)
		}
	}
}
