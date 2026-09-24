package mojev

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/weights"
)

type releasedHeadFixture struct {
	Schema         int         `json:"schema"`
	SourceRevision string      `json:"source_revision"`
	ModelSHA       string      `json:"modeling_sha256"`
	ConfigSHA      string      `json:"config_sha256"`
	WeightsSHA     string      `json:"weights_sha256"`
	WeightsSize    int64       `json:"weights_size"`
	TokenIDs       []int       `json:"token_ids"`
	State          []int       `json:"state"`
	Questions      [][]int     `json:"questions"`
	Candidates     [][][]int   `json:"candidates"`
	OptionMask     [][]int     `json:"option_mask"`
	Hidden         [][]float32 `json:"pre_norm_hidden"`
	Logits         [][]float32 `json:"logits"`
}

func pinnedReleasedHead(t *testing.T) releasedHeadFixture {
	t.Helper()
	for _, pin := range []struct{ path, sha string }{
		{"../../scripts/mojev_oracle_released_head.py", "dd86f2c5c82fd6169da933c9319529f2e1753d31ea60c716dec4a6f28444792a"},
		{"testdata/released_head.json", "32ae8f212b0a4def89cc2e95f3655ff6612c28cbd0ff5a1398d27f7873b271ff"},
	} {
		data, err := os.ReadFile(pin.path)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(data)) != pin.sha {
			t.Fatalf("pinned oracle hash changed: %s", pin.path)
		}
	}
	data, err := os.ReadFile("testdata/released_head.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture releasedHeadFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.SourceRevision != SourceRevision || fixture.ModelSHA != "a8e93f62d92c6748c5d001fef4f9516d6a74b10158d7265f53bab13f1091d458" || fixture.ConfigSHA != "1b3fb0dd8ae5a1e334b31bae1cfb2e8212231bd549884819304f29334313f4c9" || fixture.WeightsSHA != "eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50" || fixture.WeightsSize != 1710234304 {
		t.Fatal("released oracle provenance mismatch")
	}
	return fixture
}

func TestReleasedMoJevHead(t *testing.T) {
	fixture := pinnedReleasedHead(t)
	path := os.Getenv("GO_PHERENCE_MOJEV_CHECKPOINT_DIR")
	if path == "" {
		t.Skip("set GO_PHERENCE_MOJEV_CHECKPOINT_DIR to the approved hash-verified checkpoint")
	}
	// The opt-in test reads only the four head tensors; the fixture records
	// the independently executed BF16 encoder rows from the full checkpoint.
	if info, err := os.Stat(path + "/model.safetensors"); err != nil || info.Size() != fixture.WeightsSize {
		t.Fatal("wrong checkpoint size", err)
	}
	file, err := os.Open(path + "/model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hash := sha256.New()
	if _, err = io.Copy(hash, file); err != nil {
		t.Fatal(err)
	}
	file.Close()
	if fmt.Sprintf("%x", hash.Sum(nil)) != fixture.WeightsSHA {
		t.Fatal("checkpoint hash mismatch")
	}
	src, err := weights.OpenSafetensors(path)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	head, err := LoadHead(src, 1024, 512)
	if err != nil {
		t.Fatal(err)
	}
	bits := func(row []int) []bool {
		out := make([]bool, len(row))
		for i, v := range row {
			if v != 0 && v != 1 {
				t.Fatal("invalid fixture mask")
			}
			out[i] = v == 1
		}
		return out
	}
	state := bits(fixture.State)
	questions := make([][]bool, len(fixture.Questions))
	options := make([][][]bool, len(fixture.Candidates))
	optionMask := make([][]bool, len(fixture.OptionMask))
	for f, q := range fixture.Questions {
		questions[f] = bits(q)
		optionMask[f] = bits(fixture.OptionMask[f])
		options[f] = make([][]bool, len(fixture.Candidates[f]))
		for n, c := range fixture.Candidates[f] {
			options[f][n] = bits(c)
		}
	}
	hidden := make([]float32, 0, len(fixture.Hidden)*1024)
	for _, row := range fixture.Hidden {
		if len(row) != 1024 {
			t.Fatal("wrong hidden width")
		}
		hidden = append(hidden, row...)
	}
	got, err := head.ScoreHidden(hidden, state, questions, options, optionMask)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(fixture.Logits) {
		t.Fatal("wrong field count")
	}
	for f, row := range got {
		if len(row) != len(fixture.Logits[f]) {
			t.Fatal("wrong candidate count")
		}
		for n, v := range row {
			want := fixture.Logits[f][n]
			if !optionMask[f][n] {
				if math.Float32bits(v) != math.Float32bits(-math.MaxFloat32) {
					t.Fatalf("masked logit %g", v)
				}
				continue
			}
			if math.Abs(float64(v-want)) > 3e-4 {
				t.Fatalf("logit[%d][%d]=%.9g want %.9g", f, n, v, want)
			}
		}
	}
}
