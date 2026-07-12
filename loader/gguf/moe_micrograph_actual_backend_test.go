package gguf

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

const (
	moeOracleK       = 256
	moeOracleI       = 128
	moeOracleH       = 256
	moeOracleExperts = 4
	moeOracleSlots   = 2
	moeOracleTokens  = 3
)

type moeMicrographOracleFixture struct {
	Provenance struct {
		LlamaCPPRevision string `json:"llama_cpp_revision"`
		Graph            string `json:"graph"`
		BuilderSemantics string `json:"builder_semantics"`
		Backend          string `json:"backend"`
		BackendThreads   int    `json:"backend_threads"`
		WeightBufferType string `json:"weight_buffer_type"`
		LogExcerpt       string `json:"log_excerpt"`
	} `json:"provenance"`
	Shape struct {
		GateUpExps            [3]int `json:"gate_up_exps"`
		Src1                  [3]int `json:"src1"`
		IDs                   [2]int `json:"ids"`
		SelectedExpertWeights [2]int `json:"selected_expert_weights"`
		FFNMoEGateUp          [3]int `json:"ffn_moe_gate_up"`
		FFNMoEGate            [3]int `json:"ffn_moe_gate"`
		FFNMoEUp              [3]int `json:"ffn_moe_up"`
		FFNMoEGELU            [3]int `json:"ffn_moe_gelu"`
		FFNMoEAct             [3]int `json:"ffn_moe_act"`
		DownExps              [3]int `json:"down_exps"`
		FFNMoEDown            [3]int `json:"ffn_moe_down"`
		FFNMoEWeighted        [3]int `json:"ffn_moe_weighted"`
		FFNMoEReduced         [2]int `json:"ffn_moe_reduced"`
	} `json:"shape"`
	Dimensions struct {
		K            int `json:"k"`
		Intermediate int `json:"intermediate"`
		Hidden       int `json:"hidden"`
		Experts      int `json:"experts"`
		Slots        int `json:"slots"`
		Tokens       int `json:"tokens"`
	} `json:"dimensions"`
	TokenSlotIDs          [][]int32 `json:"token_slot_ids"`
	SelectedExpertWeights []float32 `json:"selected_expert_weights"`
	RuntimeRepack         struct {
		GateUpQ4 bool `json:"gate_up_q4"`
		DownQ8   bool `json:"down_q8"`
	} `json:"runtime_repack"`
	Outputs struct {
		FFNMoEGateUp   []float32 `json:"ffn_moe_gate_up"`
		FFNMoEGate     []float32 `json:"ffn_moe_gate"`
		FFNMoEUp       []float32 `json:"ffn_moe_up"`
		FFNMoEGELU     []float32 `json:"ffn_moe_gelu"`
		FFNMoEAct      []float32 `json:"ffn_moe_act"`
		FFNMoEDown     []float32 `json:"ffn_moe_down"`
		FFNMoEWeighted []float32 `json:"ffn_moe_weighted"`
		FFNMoEReduced  []float32 `json:"ffn_moe_reduced"`
	} `json:"outputs"`
}

func loadMoeMicrographActualBackendFixture(t *testing.T) (moeMicrographOracleFixture, []byte, []byte, []float32, []int32, []float32) {
	t.Helper()
	base := filepath.Join("testdata", "actual_ggml_moe_micrograph_oracle")
	blob, err := os.ReadFile(base + ".json")
	if err != nil {
		t.Fatalf("read %s.json: %v", base, err)
	}
	var fixture moeMicrographOracleFixture
	if err := json.Unmarshal(blob, &fixture); err != nil {
		t.Fatalf("unmarshal %s.json: %v", base, err)
	}
	gateUpRaw, err := os.ReadFile(base + "_gate_up_q4.bin")
	if err != nil {
		t.Fatalf("read %s_gate_up_q4.bin: %v", base, err)
	}
	downRaw, err := os.ReadFile(base + "_down_q8.bin")
	if err != nil {
		t.Fatalf("read %s_down_q8.bin: %v", base, err)
	}
	src1Raw, err := os.ReadFile(base + "_src1_f32.bin")
	if err != nil {
		t.Fatalf("read %s_src1_f32.bin: %v", base, err)
	}
	if len(src1Raw)%4 != 0 {
		t.Fatalf("src1 bytes=%d not multiple of 4", len(src1Raw))
	}
	src1 := make([]float32, len(src1Raw)/4)
	for i := range src1 {
		src1[i] = math.Float32frombits(binary.LittleEndian.Uint32(src1Raw[i*4 : i*4+4]))
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
	weightsRaw, err := os.ReadFile(base + "_selected_weights_f32.bin")
	if err != nil {
		t.Fatalf("read %s_selected_weights_f32.bin: %v", base, err)
	}
	if len(weightsRaw)%4 != 0 {
		t.Fatalf("selected weight bytes=%d not multiple of 4", len(weightsRaw))
	}
	weights := make([]float32, len(weightsRaw)/4)
	for i := range weights {
		weights[i] = math.Float32frombits(binary.LittleEndian.Uint32(weightsRaw[i*4 : i*4+4]))
	}
	return fixture, gateUpRaw, downRaw, src1, ids, weights
}

func TestActualBackendMoEMicrographOracle(t *testing.T) {
	fixture, gateUpRaw, downRaw, src1, ids, selectedWeights := loadMoeMicrographActualBackendFixture(t)
	if fixture.Provenance.LlamaCPPRevision != "4a6735f1cf0594250958bcc839267696c7b998a4" ||
		fixture.Provenance.Graph != "ggml_moe_micrograph" || fixture.Provenance.Backend != "CPU" ||
		fixture.Provenance.BackendThreads != 1 || fixture.Provenance.WeightBufferType != "ggml_backend_cpu_repack_buffer_type" {
		t.Fatalf("fixture provenance mismatch %+v", fixture.Provenance)
	}
	if !fixture.RuntimeRepack.GateUpQ4 || fixture.RuntimeRepack.DownQ8 || !strings.Contains(fixture.Provenance.LogExcerpt, "repack tensor gate_up_q4 with q4_K_8x8") {
		t.Fatalf("fixture repack provenance mismatch runtime=%+v log=%q", fixture.RuntimeRepack, fixture.Provenance.LogExcerpt)
	}
	if !strings.Contains(fixture.Provenance.BuilderSemantics, "merged gate_up path") || !strings.Contains(fixture.Provenance.BuilderSemantics, "ggml_gelu(gate) * up") {
		t.Fatalf("fixture builder semantics mismatch %q", fixture.Provenance.BuilderSemantics)
	}
	if fixture.Shape.GateUpExps != [3]int{moeOracleK, 2 * moeOracleI, moeOracleExperts} ||
		fixture.Shape.Src1 != [3]int{moeOracleK, moeOracleSlots, moeOracleTokens} ||
		fixture.Shape.IDs != [2]int{moeOracleSlots, moeOracleTokens} ||
		fixture.Shape.SelectedExpertWeights != [2]int{moeOracleSlots, moeOracleTokens} ||
		fixture.Shape.DownExps != [3]int{moeOracleI, moeOracleH, moeOracleExperts} ||
		fixture.Shape.FFNMoEGateUp != [3]int{2 * moeOracleI, moeOracleSlots, moeOracleTokens} ||
		fixture.Shape.FFNMoEGate != [3]int{moeOracleI, moeOracleSlots, moeOracleTokens} ||
		fixture.Shape.FFNMoEUp != [3]int{moeOracleI, moeOracleSlots, moeOracleTokens} ||
		fixture.Shape.FFNMoEGELU != [3]int{moeOracleI, moeOracleSlots, moeOracleTokens} ||
		fixture.Shape.FFNMoEAct != [3]int{moeOracleI, moeOracleSlots, moeOracleTokens} ||
		fixture.Shape.FFNMoEDown != [3]int{moeOracleH, moeOracleSlots, moeOracleTokens} ||
		fixture.Shape.FFNMoEWeighted != [3]int{moeOracleH, moeOracleSlots, moeOracleTokens} ||
		fixture.Shape.FFNMoEReduced != [2]int{moeOracleH, moeOracleTokens} {
		t.Fatalf("fixture shape mismatch %+v", fixture.Shape)
	}
	if fixture.Dimensions.K != moeOracleK || fixture.Dimensions.Intermediate != moeOracleI || fixture.Dimensions.Hidden != moeOracleH || fixture.Dimensions.Experts != moeOracleExperts || fixture.Dimensions.Slots != moeOracleSlots || fixture.Dimensions.Tokens != moeOracleTokens {
		t.Fatalf("fixture dimensions mismatch %+v", fixture.Dimensions)
	}
	if len(ids) != moeOracleSlots*moeOracleTokens || len(selectedWeights) != moeOracleSlots*moeOracleTokens || len(src1) != moeOracleK*moeOracleSlots*moeOracleTokens {
		t.Fatalf("fixture raw sizes ids=%d weights=%d src1=%d", len(ids), len(selectedWeights), len(src1))
	}
	if ids[0] != 2 || ids[1] != 0 || ids[2] != 1 || ids[3] != 2 || ids[4] != 2 || ids[5] != 1 {
		t.Fatalf("fixture no longer proves repeated out-of-order experts: ids=%v", ids)
	}

	q4RowBytes, err := TensorRawBytes(QuantQ4_K, moeOracleK)
	if err != nil {
		t.Fatalf("q4 row bytes: %v", err)
	}
	q8RowBytes, err := TensorRawBytes(QuantQ8_0, moeOracleI)
	if err != nil {
		t.Fatalf("q8 row bytes: %v", err)
	}
	if len(gateUpRaw) != q4RowBytes*(2*moeOracleI)*moeOracleExperts {
		t.Fatalf("gate_up_q4 bytes=%d want=%d", len(gateUpRaw), q4RowBytes*(2*moeOracleI)*moeOracleExperts)
	}
	if len(downRaw) != q8RowBytes*moeOracleH*moeOracleExperts {
		t.Fatalf("down_q8 bytes=%d want=%d", len(downRaw), q8RowBytes*moeOracleH*moeOracleExperts)
	}

	gateUpTiles := make([][]*experimentalQ4K8x8Tile, moeOracleExperts)
	for expert := 0; expert < moeOracleExperts; expert++ {
		gateUpTiles[expert] = make([]*experimentalQ4K8x8Tile, (2*moeOracleI)/experimentalQ4K8x8Rows)
		for tileIdx := range gateUpTiles[expert] {
			start := expert*(2*moeOracleI)*q4RowBytes + tileIdx*experimentalQ4K8x8Rows*q4RowBytes
			tile, err := newExperimentalQ4K8x8Tile(gateUpRaw[start:start+experimentalQ4K8x8Rows*q4RowBytes], moeOracleK)
			if err != nil {
				t.Fatalf("expert %d tile %d repack gate_up: %v", expert, tileIdx, err)
			}
			gateUpTiles[expert][tileIdx] = tile
		}
	}

	gateUp := make([]float32, moeOracleTokens*moeOracleSlots*2*moeOracleI)
	gate := make([]float32, moeOracleTokens*moeOracleSlots*moeOracleI)
	up := make([]float32, moeOracleTokens*moeOracleSlots*moeOracleI)
	geluGate := make([]float32, moeOracleTokens*moeOracleSlots*moeOracleI)
	act := make([]float32, moeOracleTokens*moeOracleSlots*moeOracleI)
	down := make([]float32, moeOracleTokens*moeOracleSlots*moeOracleH)
	weighted := make([]float32, len(down))
	reduced := make([]float32, moeOracleTokens*moeOracleH)

	for token := 0; token < moeOracleTokens; token++ {
		for slot := 0; slot < moeOracleSlots; slot++ {
			pos := token*moeOracleSlots + slot
			expert := int(ids[pos])
			if expert < 0 || expert >= moeOracleExperts {
				t.Fatalf("ids[%d,%d]=%d outside experts=%d", token, slot, expert, moeOracleExperts)
			}
			actIn := src1[pos*moeOracleK : (pos+1)*moeOracleK]
			for tileIdx, tile := range gateUpTiles[expert] {
				out, err := tile.mulF32Activation(actIn)
				if err != nil {
					t.Fatalf("expert %d token=%d slot=%d gate_up tile=%d: %v", expert, token, slot, tileIdx, err)
				}
				copy(gateUp[pos*(2*moeOracleI)+tileIdx*experimentalQ4K8x8Rows:], out)
			}
			copy(gate[pos*moeOracleI:(pos+1)*moeOracleI], gateUp[pos*(2*moeOracleI):pos*(2*moeOracleI)+moeOracleI])
			copy(up[pos*moeOracleI:(pos+1)*moeOracleI], gateUp[pos*(2*moeOracleI)+moeOracleI:(pos+1)*(2*moeOracleI)])
			for i := 0; i < moeOracleI; i++ {
				g := simd.GELUTanhScalar(gate[pos*moeOracleI+i])
				geluGate[pos*moeOracleI+i] = g
				act[pos*moeOracleI+i] = g * up[pos*moeOracleI+i]
			}
			q8, err := QuantizeQ8_0(act[pos*moeOracleI : (pos+1)*moeOracleI])
			if err != nil {
				t.Fatalf("token=%d slot=%d quantize act q8_0: %v", token, slot, err)
			}
			expertOff := expert * moeOracleH * q8RowBytes
			for row := 0; row < moeOracleH; row++ {
				rowRaw := downRaw[expertOff+row*q8RowBytes : expertOff+(row+1)*q8RowBytes]
				v, err := DotQ8_0Q8_0(rowRaw, q8, moeOracleI)
				if err != nil {
					t.Fatalf("expert %d token=%d slot=%d down row=%d: %v", expert, token, slot, row, err)
				}
				down[pos*moeOracleH+row] = v
				weighted[pos*moeOracleH+row] = v * selectedWeights[pos]
				reduced[token*moeOracleH+row] += weighted[pos*moeOracleH+row]
			}
		}
	}

	assertOracleBoundary(t, "ffn_moe_gate_up", gateUp, fixture.Outputs.FFNMoEGateUp, 2*moeOracleI, moeOracleSlots, moeOracleTokens, 1e-4)
	// gate and up are strided views; validate their materialized downstream results.
	assertOracleBoundary(t, "ffn_moe_gelu", geluGate, fixture.Outputs.FFNMoEGELU, moeOracleI, moeOracleSlots, moeOracleTokens, 3e-4)
	assertOracleBoundary(t, "ffn_moe_act", act, fixture.Outputs.FFNMoEAct, moeOracleI, moeOracleSlots, moeOracleTokens, 1e-4)
	assertOracleBoundary(t, "ffn_moe_down", down, fixture.Outputs.FFNMoEDown, moeOracleH, moeOracleSlots, moeOracleTokens, 2e-4)
	assertOracleBoundary(t, "ffn_moe_weighted", weighted, fixture.Outputs.FFNMoEWeighted, moeOracleH, moeOracleSlots, moeOracleTokens, 2e-4)
	assertOracleReducedBoundary(t, reduced, fixture.Outputs.FFNMoEReduced, moeOracleH, moeOracleTokens, 2e-4)
}

func assertOracleBoundary(t *testing.T, name string, got, want []float32, rows, slots, tokens int, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len=%d want=%d", name, len(got), len(want))
	}
	for token := 0; token < tokens; token++ {
		for slot := 0; slot < slots; slot++ {
			base := (token*slots + slot) * rows
			for row := 0; row < rows; row++ {
				i := base + row
				delta := math.Abs(float64(got[i] - want[i]))
				if delta > tol {
					t.Fatalf("%s first mismatch token=%d slot=%d row=%d got=%.9g want=%.9g delta=%.9g tol=%.9g", name, token, slot, row, got[i], want[i], delta, tol)
				}
			}
		}
	}
}

func assertOracleReducedBoundary(t *testing.T, got, want []float32, rows, tokens int, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("ffn_moe_reduced len=%d want=%d", len(got), len(want))
	}
	for token := 0; token < tokens; token++ {
		base := token * rows
		for row := 0; row < rows; row++ {
			i := base + row
			delta := math.Abs(float64(got[i] - want[i]))
			if delta > tol {
				t.Fatalf("ffn_moe_reduced first mismatch token=%d row=%d got=%.9g want=%.9g delta=%.9g tol=%.9g", token, row, got[i], want[i], delta, tol)
			}
		}
	}
}
