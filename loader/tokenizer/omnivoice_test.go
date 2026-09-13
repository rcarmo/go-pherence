package tokenizer

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestOmniVoiceRealTokenizerParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_REAL_OMNIVOICE")
	if path == "" {
		t.Skip("set model dir")
	}
	tok, err := Load(path + "/tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("../../testdata/omnivoice/tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Cases []struct {
			Text string
			IDs  []int
		}
	}
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if len(f.Cases) < 5 {
		t.Fatal("no cases")
	}
	for _, c := range f.Cases {
		if got := tok.Encode(c.Text); !reflect.DeepEqual(got, c.IDs) {
			t.Errorf("%q\ngot %v\nwant %v", c.Text, got, c.IDs)
		}
	}
}
func TestSingleDigitPreTokenization(t *testing.T) {
	raw := `{"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"` + "" + `"}}]},"model":{"vocab":{},"merges":[]}}`
	_ = raw
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
