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
	q4KGEMVOracleK = 2816
	q4KGEMVOracleM = 8
	q4KGEMVOracleN = 1
)

type q4KGEMVOracleFixture struct {
	Shape                [2]int    `json:"shape"`
	K                    int       `json:"k"`
	RuntimeRepack        bool      `json:"runtime_repack"`
	MulMatTokens         int       `json:"mul_mat_tokens"`
	Output               []float32 `json:"output"`
	Vecdot               []float32 `json:"vecdot"`
	DirectGEMV           []float32 `json:"direct_gemv"`
	MaxAbsDiff           float32   `json:"max_abs_diff"`
	MaxAbsDiffDirectGEMV float32   `json:"max_abs_diff_direct_gemv"`
}

func loadQ4KGEMVActualBackendFixture(t *testing.T) (q4KGEMVOracleFixture, []byte, []float32) {
	t.Helper()
	base := filepath.Join("testdata", "actual_ggml_q4k_gemv_oracle")
	blob, err := os.ReadFile(base + ".json")
	if err != nil {
		t.Fatalf("read %s.json: %v", base, err)
	}
	var fixture q4KGEMVOracleFixture
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
	return fixture, q4Raw, acts
}

func TestQ4K8x8ActualBackendGEMV(t *testing.T) {
	fixture, q4Raw, acts := loadQ4KGEMVActualBackendFixture(t)
	if fixture.Shape != [2]int{q4KGEMVOracleM, q4KGEMVOracleN} || fixture.K != q4KGEMVOracleK {
		t.Fatalf("fixture shape/k mismatch shape=%v k=%d", fixture.Shape, fixture.K)
	}
	if !fixture.RuntimeRepack || fixture.MulMatTokens != 1 {
		t.Fatalf("fixture does not prove repacked single-column mul_mat: runtime_repack=%v mul_mat_tokens=%d", fixture.RuntimeRepack, fixture.MulMatTokens)
	}
	tile, err := newExperimentalQ4K8x8Tile(q4Raw, fixture.K)
	if err != nil {
		t.Fatalf("repack q4 tile: %v", err)
	}
	out, err := tile.mulF32Activation(acts)
	if err != nil {
		t.Fatalf("mul f32 gemv tile: %v", err)
	}
	maxDelta := float32(0)
	maxVecdotGap := float32(0)
	maxDirectGEMVGap := float32(0)
	oracleVecdotGap := float32(0)
	distinctCount := 0
	for i := 0; i < q4KGEMVOracleM; i++ {
		got := out[i]
		delta := float32(math.Abs(float64(got - fixture.Output[i])))
		if delta > maxDelta {
			maxDelta = delta
		}
		if delta > 1e-4 {
			t.Fatalf("output[%d]=%.9g want %.9g delta %.9g", i, got, fixture.Output[i], delta)
		}
		if directDelta := float32(math.Abs(float64(got - fixture.DirectGEMV[i]))); directDelta > maxDirectGEMVGap {
			maxDirectGEMVGap = directDelta
		}
		gap := float32(math.Abs(float64(fixture.Output[i] - fixture.Vecdot[i])))
		if gap > oracleVecdotGap {
			oracleVecdotGap = gap
		}
		gotVecdotGap := float32(math.Abs(float64(got - fixture.Vecdot[i])))
		if gotVecdotGap > maxVecdotGap {
			maxVecdotGap = gotVecdotGap
		}
		if gap > 5e-7 && gotVecdotGap > gap*0.5 {
			distinctCount++
		}
	}
	if distinctCount == 0 {
		t.Fatalf("computed tile collapsed to vecdot path: max oracle-vs-vecdot gap %.9g, max got-vs-vecdot gap %.9g", oracleVecdotGap, maxVecdotGap)
	}
	t.Logf("Q4_K 8x8 actual-backend GEMV max delta: %.9g", maxDelta)
	t.Logf("Q4_K 8x8 Go-vs-direct-gemv max gap: %.9g", maxDirectGEMVGap)
	t.Logf("Q4_K 8x8 GEMV oracle-vs-vecdot max gap: %.9g; distinguishable outputs: %d", oracleVecdotGap, distinctCount)
}
