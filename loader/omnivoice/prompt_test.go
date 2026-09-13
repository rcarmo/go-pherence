package omnivoice

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestRealPreparedPromptParity(t *testing.T) {
	root := os.Getenv("GO_PHERENCE_REAL_OMNIVOICE")
	if root == "" {
		t.Skip("set GO_PHERENCE_REAL_OMNIVOICE")
	}
	fixture := os.Getenv("GO_PHERENCE_REAL_PROMPT")
	if fixture == "" {
		t.Skip("set GO_PHERENCE_REAL_PROMPT to private reference-export fixture")
	}
	raw, err := os.ReadFile(fixture)
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Reference CachedReferenceTokens
		Expected  PreparedPrompt
	}
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig(root)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.Load(filepath.Join(root, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	got, err := PrepareInferenceInputs(cfg, tok, f.Expected.Text, f.Expected.TargetFrames, f.Reference, PreparePromptOptions{Language: "en", Denoise: true})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, f.Expected) {
		t.Fatalf("prepared prompt differs from upstream fixture (got %d tokens, want %d)", got.Conditional.Tokens, f.Expected.Conditional.Tokens)
	}
}

func TestCachedReferenceValidation(t *testing.T) {
	good := CachedReferenceTokens{Books: 1, Frames: 1, Codes: []int{0}, Transcript: "Reference."}
	if err := good.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, r := range []CachedReferenceTokens{{Books: math.MaxInt, Frames: 2}, {Books: 1, Frames: 501}, {Books: 1, Frames: 1, Codes: []int{0}}, {Books: 1, Frames: 2, Codes: []int{0}, Transcript: "x"}} {
		if r.Validate() == nil {
			t.Fatal("accepted invalid reference")
		}
	}
	for _, v := range []float64{math.NaN(), math.Inf(1), -1} {
		r := good
		r.RefRMS = &v
		if r.Validate() == nil {
			t.Fatal("accepted invalid RMS")
		}
	}
}

func TestCombineText(t *testing.T) {
	for _, c := range [][3]string{{" Captain. ", " Vulcans never bluff. ", "Vulcans never bluff. Captain."}, {"你 好 （是）\n", "参 考\t", "参考你好(是)"}, {"a\r\nb\t c", "", "ab c"}} {
		if got := combineText(c[0], c[1]); got != c[2] {
			t.Fatalf("%q != %q", got, c[2])
		}
	}
}
