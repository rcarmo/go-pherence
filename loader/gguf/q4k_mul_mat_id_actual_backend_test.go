package gguf

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

const (
	q4KMulMatIDOracleK       = 256
	q4KMulMatIDOracleOut     = 8
	q4KMulMatIDOracleExperts = 4
	q4KMulMatIDOracleNIDs    = 2
	q4KMulMatIDOracleTokens  = 3
)

type q4KMulMatIDOracleFixture struct {
	Provenance struct {
		LlamaCPPRevision string `json:"llama_cpp_revision"`
		Graph            string `json:"graph"`
		Backend          string `json:"backend"`
		BackendThreads   int    `json:"backend_threads"`
		WeightBufferType string `json:"weight_buffer_type"`
		LogExcerpt       string `json:"log_excerpt"`
	} `json:"provenance"`
	Shape struct {
		Src0As [3]int `json:"src0_as"`
		Src1B  [3]int `json:"src1_b"`
		IDs    [2]int `json:"ids"`
		Dst    [3]int `json:"dst"`
	} `json:"shape"`
	Layout struct {
		Src0As string `json:"src0_as"`
		Src1B  string `json:"src1_b"`
		IDs    string `json:"ids"`
		Dst    string `json:"dst"`
	} `json:"layout"`
	Dimensions struct {
		K       int `json:"k"`
		Out     int `json:"out"`
		Experts int `json:"experts"`
		NIDs    int `json:"n_ids"`
		Tokens  int `json:"tokens"`
	} `json:"dimensions"`
	ExpertScanOrder string    `json:"expert_scan_order"`
	TokenSlotIDs    [][]int32 `json:"token_slot_ids"`
	RuntimeRepack   bool      `json:"runtime_repack"`
	Output          []float32 `json:"output"`
	ExpectedVecdot  []float32 `json:"expected_vecdot"`
	MaxAbsDiff      float32   `json:"max_abs_diff"`
}

func loadQ4KMulMatIDActualBackendFixture(t *testing.T) (q4KMulMatIDOracleFixture, []byte, []float32, []int32) {
	t.Helper()
	base := filepath.Join("testdata", "actual_ggml_mul_mat_id_oracle")
	blob, err := os.ReadFile(base + ".json")
	if err != nil {
		t.Fatalf("read %s.json: %v", base, err)
	}
	var fixture q4KMulMatIDOracleFixture
	if err := json.Unmarshal(blob, &fixture); err != nil {
		t.Fatalf("unmarshal %s.json: %v", base, err)
	}
	q4Raw, err := os.ReadFile(base + "_q4.bin")
	if err != nil {
		t.Fatalf("read %s_q4.bin: %v", base, err)
	}
	actRaw, err := os.ReadFile(base + "_act_f32.bin")
	if err != nil {
		t.Fatalf("read %s_act_f32.bin: %v", base, err)
	}
	if len(actRaw)%4 != 0 {
		t.Fatalf("activation bytes=%d not multiple of 4", len(actRaw))
	}
	acts := make([]float32, len(actRaw)/4)
	for i := range acts {
		acts[i] = math.Float32frombits(binary.LittleEndian.Uint32(actRaw[i*4 : i*4+4]))
	}
	idsRaw, err := os.ReadFile(base + "_ids_i32.bin")
	if err != nil {
		t.Fatalf("read %s_ids_i32.bin: %v", base, err)
	}
	if len(idsRaw)%4 != 0 {
		t.Fatalf("ids bytes=%d not multiple of 4", len(idsRaw))
	}
	ids := make([]int32, len(idsRaw)/4)
	for i := range ids {
		ids[i] = int32(binary.LittleEndian.Uint32(idsRaw[i*4 : i*4+4]))
	}
	return fixture, q4Raw, acts, ids
}

func TestQ4K8x8ActualBackendMulMatIDGrouping(t *testing.T) {
	fixture, q4Raw, acts, ids := loadQ4KMulMatIDActualBackendFixture(t)
	if fixture.Shape.Src0As != [3]int{q4KMulMatIDOracleK, q4KMulMatIDOracleOut, q4KMulMatIDOracleExperts} ||
		fixture.Shape.Src1B != [3]int{q4KMulMatIDOracleK, q4KMulMatIDOracleNIDs, q4KMulMatIDOracleTokens} ||
		fixture.Shape.IDs != [2]int{q4KMulMatIDOracleNIDs, q4KMulMatIDOracleTokens} ||
		fixture.Shape.Dst != [3]int{q4KMulMatIDOracleOut, q4KMulMatIDOracleNIDs, q4KMulMatIDOracleTokens} {
		t.Fatalf("fixture shape mismatch src0=%v src1=%v ids=%v dst=%v", fixture.Shape.Src0As, fixture.Shape.Src1B, fixture.Shape.IDs, fixture.Shape.Dst)
	}
	if fixture.Dimensions.K != q4KMulMatIDOracleK || fixture.Dimensions.Out != q4KMulMatIDOracleOut || fixture.Dimensions.Experts != q4KMulMatIDOracleExperts || fixture.Dimensions.NIDs != q4KMulMatIDOracleNIDs || fixture.Dimensions.Tokens != q4KMulMatIDOracleTokens {
		t.Fatalf("fixture dimensions mismatch %+v", fixture.Dimensions)
	}
	if !fixture.RuntimeRepack || fixture.Provenance.Graph != "ggml_mul_mat_id" || fixture.Provenance.Backend != "CPU" || fixture.Provenance.BackendThreads != 1 {
		t.Fatalf("fixture provenance mismatch %+v runtime_repack=%v", fixture.Provenance, fixture.RuntimeRepack)
	}
	if got, want := len(fixture.TokenSlotIDs), q4KMulMatIDOracleTokens; got != want {
		t.Fatalf("token_slot_ids tokens=%d want=%d", got, want)
	}
	for token, row := range fixture.TokenSlotIDs {
		if len(row) != q4KMulMatIDOracleNIDs {
			t.Fatalf("token_slot_ids[%d] len=%d want=%d", token, len(row), q4KMulMatIDOracleNIDs)
		}
	}
	if len(ids) != q4KMulMatIDOracleNIDs*q4KMulMatIDOracleTokens {
		t.Fatalf("ids len=%d want=%d", len(ids), q4KMulMatIDOracleNIDs*q4KMulMatIDOracleTokens)
	}
	if ids[0] != 2 || ids[1] != 0 || ids[2] != 1 || ids[3] != 2 || ids[4] != 2 || ids[5] != 1 {
		t.Fatalf("fixture no longer proves repeated out-of-order experts: ids=%v", ids)
	}
	if len(acts) != q4KMulMatIDOracleK*q4KMulMatIDOracleNIDs*q4KMulMatIDOracleTokens {
		t.Fatalf("acts len=%d want=%d", len(acts), q4KMulMatIDOracleK*q4KMulMatIDOracleNIDs*q4KMulMatIDOracleTokens)
	}
	if len(fixture.Output) != q4KMulMatIDOracleOut*q4KMulMatIDOracleNIDs*q4KMulMatIDOracleTokens {
		t.Fatalf("output len=%d want=%d", len(fixture.Output), q4KMulMatIDOracleOut*q4KMulMatIDOracleNIDs*q4KMulMatIDOracleTokens)
	}
	if len(fixture.ExpectedVecdot) != len(fixture.Output) {
		t.Fatalf("expected_vecdot len=%d want=%d", len(fixture.ExpectedVecdot), len(fixture.Output))
	}

	rowBytes, err := TensorRawBytes(QuantQ4_K, q4KMulMatIDOracleK)
	if err != nil {
		t.Fatalf("row bytes: %v", err)
	}
	expertBytes := rowBytes * q4KMulMatIDOracleOut
	if len(q4Raw) != expertBytes*q4KMulMatIDOracleExperts {
		t.Fatalf("q4 bytes=%d want=%d", len(q4Raw), expertBytes*q4KMulMatIDOracleExperts)
	}
	tiles := make([]*experimentalQ4K8x8Tile, q4KMulMatIDOracleExperts)
	for expert := 0; expert < q4KMulMatIDOracleExperts; expert++ {
		tile, err := newExperimentalQ4K8x8Tile(q4Raw[expert*expertBytes:(expert+1)*expertBytes], q4KMulMatIDOracleK)
		if err != nil {
			t.Fatalf("expert %d repack q4 tile: %v", expert, err)
		}
		tiles[expert] = tile
	}

	type tokenSlot struct {
		token int
		slot  int
	}
	grouped := make([][]tokenSlot, q4KMulMatIDOracleExperts)
	for token := 0; token < q4KMulMatIDOracleTokens; token++ {
		for slot := 0; slot < q4KMulMatIDOracleNIDs; slot++ {
			expert := int(ids[token*q4KMulMatIDOracleNIDs+slot])
			if expert < 0 || expert >= q4KMulMatIDOracleExperts {
				t.Fatalf("ids[%d,%d]=%d outside experts=%d", token, slot, expert, q4KMulMatIDOracleExperts)
			}
			grouped[expert] = append(grouped[expert], tokenSlot{token: token, slot: slot})
		}
	}
	if len(grouped[2]) < 3 || len(grouped[3]) != 0 {
		t.Fatalf("fixture grouping does not prove repeats/unused expert: lens=%d,%d,%d,%d", len(grouped[0]), len(grouped[1]), len(grouped[2]), len(grouped[3]))
	}

	reproduced := make([]float32, len(fixture.Output))
	for expert, pairs := range grouped {
		if len(pairs) == 0 {
			continue
		}
		tile := tiles[expert]
		for _, pair := range pairs {
			act := acts[(pair.token*q4KMulMatIDOracleNIDs+pair.slot)*q4KMulMatIDOracleK : (pair.token*q4KMulMatIDOracleNIDs+pair.slot+1)*q4KMulMatIDOracleK]
			out, err := tile.mulF32Activation(act)
			if err != nil {
				t.Fatalf("expert %d token=%d slot=%d mul f32 activation: %v", expert, pair.token, pair.slot, err)
			}
			copy(reproduced[(pair.token*q4KMulMatIDOracleNIDs+pair.slot)*q4KMulMatIDOracleOut:], out)
		}
	}

	maxDelta := float32(0)
	for i, got := range reproduced {
		delta := float32(math.Abs(float64(got - fixture.Output[i])))
		if delta > maxDelta {
			maxDelta = delta
		}
		if delta > 1e-5 {
			t.Fatalf("output[%d]=%.9g want %.9g delta %.9g", i, got, fixture.Output[i], delta)
		}
		vecdotDelta := float32(math.Abs(float64(got - fixture.ExpectedVecdot[i])))
		if vecdotDelta > 1e-5 {
			t.Fatalf("vecdot[%d]=%.9g want %.9g delta %.9g", i, got, fixture.ExpectedVecdot[i], vecdotDelta)
		}
	}
	t.Logf("Q4_K 8x8 actual-backend MUL_MAT_ID max delta: %.9g", maxDelta)
}
