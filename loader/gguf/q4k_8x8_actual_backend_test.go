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
	q4KOracleK = 2816
	q4KOracleM = 8
	q4KOracleN = 8
)

type q4KOracleFixture struct {
	Shape      [2]int      `json:"shape"`
	K          int         `json:"k"`
	Output     [][]float32 `json:"output"`
	Vecdot     [][]float32 `json:"vecdot"`
	MaxAbsDiff float32     `json:"max_abs_diff"`
}

func loadQ4K8x8ActualBackendFixture(t *testing.T) (q4KOracleFixture, []byte, []float32) {
	t.Helper()
	base := filepath.Join("testdata", "actual_ggml_q4k_oracle")
	blob, err := os.ReadFile(base + ".json")
	if err != nil {
		t.Fatalf("read %s.json: %v", base, err)
	}
	var fixture q4KOracleFixture
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

func TestQ4K8x8ActualBackendTile(t *testing.T) {
	fixture, q4Raw, acts := loadQ4K8x8ActualBackendFixture(t)
	if fixture.Shape != [2]int{q4KOracleM, q4KOracleN} || fixture.K != q4KOracleK {
		t.Fatalf("fixture shape/k mismatch shape=%v k=%d", fixture.Shape, fixture.K)
	}
	tile, err := newExperimentalQ4K8x8Tile(q4Raw, fixture.K)
	if err != nil {
		t.Fatalf("repack q4 tile: %v", err)
	}
	out, err := tile.mulF32ActivationRows(acts)
	if err != nil {
		t.Fatalf("mul f32 tile: %v", err)
	}
	maxDelta := float32(0)
	maxVecdotGap := float32(0)
	oracleVecdotGap := float32(0)
	distinctCount := 0
	for i := 0; i < q4KOracleM; i++ {
		for j := 0; j < q4KOracleN; j++ {
			got := out[i*q4KOracleN+j]
			delta := float32(math.Abs(float64(got - fixture.Output[i][j])))
			if delta > maxDelta {
				maxDelta = delta
			}
			if delta > 1e-4 {
				t.Fatalf("output[%d][%d]=%.9g want %.9g delta %.9g", i, j, got, fixture.Output[i][j], delta)
			}
			gap := float32(math.Abs(float64(fixture.Output[i][j] - fixture.Vecdot[i][j])))
			if gap > oracleVecdotGap {
				oracleVecdotGap = gap
			}
			gotVecdotGap := float32(math.Abs(float64(got - fixture.Vecdot[i][j])))
			if gotVecdotGap > maxVecdotGap {
				maxVecdotGap = gotVecdotGap
			}
			if gap > 5e-7 && gotVecdotGap > gap*0.5 {
				distinctCount++
			}
		}
	}
	if distinctCount == 0 {
		t.Fatalf("computed tile collapsed to vecdot path: max oracle-vs-vecdot gap %.9g, max got-vs-vecdot gap %.9g", oracleVecdotGap, maxVecdotGap)
	}
	t.Logf("Q4_K 8x8 actual-backend max delta: %.9g", maxDelta)
	t.Logf("Q4_K 8x8 oracle-vs-vecdot max gap: %.9g; distinguishable cells: %d", oracleVecdotGap, distinctCount)
}
