package diffusiongemma

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func syntheticCPUGroupedExpertIndex(t *testing.T) *GGUFExpertIndex {
	t.Helper()
	const experts = 3
	gateUp := &gguf.ExpertMatrices{Name: "synthetic.gate_up", QType: gguf.QuantQ4_K, InDim: 256, OutDim: 64, Experts: experts}
	gateRowBytes, err := gateUp.RowBytes()
	if err != nil {
		t.Fatal(err)
	}
	gateUp.Raw = make([]byte, gateRowBytes*gateUp.OutDim*gateUp.Experts)
	for e := 0; e < experts; e++ {
		for r := 0; r < gateUp.OutDim; r++ {
			row := gateUp.Raw[(e*gateUp.OutDim+r)*gateRowBytes : (e*gateUp.OutDim+r+1)*gateRowBytes]
			blk := row[:144]
			binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(0.02+float32((e+r)%5)*0.003))
			binary.LittleEndian.PutUint16(blk[2:4], half.F32ToF16(0.004+float32((e+r)%3)*0.001))
			for i := 0; i < 12; i++ {
				blk[4+i] = byte(1 + (i+e+r)%17)
			}
			for i := 0; i < 128; i++ {
				blk[16+i] = byte((i*7 + e*11 + r*13) & 0xff)
			}
		}
	}
	down := &gguf.ExpertMatrices{Name: "synthetic.down", QType: gguf.QuantQ8_0, InDim: 32, OutDim: 256, Experts: experts}
	downRowBytes, err := down.RowBytes()
	if err != nil {
		t.Fatal(err)
	}
	down.Raw = make([]byte, downRowBytes*down.OutDim*down.Experts)
	for e := 0; e < experts; e++ {
		for r := 0; r < down.OutDim; r++ {
			row := down.Raw[(e*down.OutDim+r)*downRowBytes : (e*down.OutDim+r+1)*downRowBytes]
			binary.LittleEndian.PutUint16(row[0:2], half.F32ToF16(0.03+float32((e+r)%7)*0.002))
			for i := 0; i < 32; i++ {
				row[2+i] = byte(int8((i+e*3+r*5)%15 - 7))
			}
		}
	}
	return &GGUFExpertIndex{
		NumLayers:    1,
		NumExperts:   experts,
		Intermediate: 32,
		HiddenSize:   256,
		entries: []ggufLayerExperts{{
			gateUp:    gateUp,
			down:      down,
			downScale: []float32{1, 0.75, 1.25},
		}},
	}
}

func syntheticGroupedExpertWork(t *testing.T, idx *GGUFExpertIndex) SelectedExpertGroupedArrays {
	t.Helper()
	items := []SelectedExpertWorkItem{
		{Position: 0, Expert: 0, Slot: 0, Weight: 0.50},
		{Position: 0, Expert: 1, Slot: 1, Weight: 0.25},
		{Position: 1, Expert: 2, Slot: 0, Weight: 0.75},
		{Position: 2, Expert: 1, Slot: 0, Weight: 0.60},
		{Position: 2, Expert: 0, Slot: 1, Weight: 0.40},
	}
	arr := BuildSelectedExpertWorkArrays(items)
	grouped, err := BuildSelectedExpertGroupedWork(arr, idx.NumExperts)
	if err != nil {
		t.Fatal(err)
	}
	ga, err := BuildSelectedExpertGroupedArrays(arr, grouped)
	if err != nil {
		t.Fatal(err)
	}
	if err := ga.ApplyDownScalesByExpert(idx.entries[0].downScale); err != nil {
		t.Fatal(err)
	}
	return ga
}

func TestRunGGUFCPUExpertsGroupedNoPostNormMatchesSingleExpertMLP(t *testing.T) {
	idx := syntheticCPUGroupedExpertIndex(t)
	ga := syntheticGroupedExpertWork(t, idx)
	positions := 3
	normedRows := make([]float32, positions*idx.HiddenSize)
	for i := range normedRows {
		normedRows[i] = float32((i%19)-9) * 0.01
	}
	gotScratch := ForwardScratch{MoeOut: make([]float32, len(normedRows))}
	if err := runGGUFCPUExpertsGroupedNoPostNorm(LayerOp{Layer: 0, Kind: OpExperts}, gotScratch, idx, normedRows, ga); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, len(normedRows))
	tmp := make([]float32, idx.HiddenSize)
	for groupIdx, expert := range ga.ActiveExperts {
		for wi := ga.Offsets[groupIdx]; wi < ga.Offsets[groupIdx+1]; wi++ {
			pos := ga.WorkPositions[wi]
			for i := range tmp {
				tmp[i] = 0
			}
			if err := idx.RunGGUFExpertMLP(0, expert, normedRows[pos*idx.HiddenSize:(pos+1)*idx.HiddenSize], tmp); err != nil {
				t.Fatal(err)
			}
			w := ga.WorkWeights[wi]
			for i, v := range tmp {
				want[pos*idx.HiddenSize+i] += w * v
			}
		}
	}
	for i := range want {
		if math.Abs(float64(gotScratch.MoeOut[i]-want[i])) > 2e-3 {
			t.Fatalf("out[%d]=%g want %g", i, gotScratch.MoeOut[i], want[i])
		}
	}
}

func TestRunGGUFCPUExpertsGroupedNoPostNormSplitRecombines(t *testing.T) {
	idx := syntheticCPUGroupedExpertIndex(t)
	ga := syntheticGroupedExpertWork(t, idx)
	positions := 3
	normedRows := make([]float32, positions*idx.HiddenSize)
	for i := range normedRows {
		normedRows[i] = float32((i%23)-11) * 0.0125
	}
	full := ForwardScratch{MoeOut: make([]float32, len(normedRows))}
	if err := runGGUFCPUExpertsGroupedNoPostNorm(LayerOp{Layer: 0, Kind: OpExperts}, full, idx, normedRows, ga); err != nil {
		t.Fatal(err)
	}
	kept, dropped, err := SplitSelectedExpertGroupedArrays(ga, func(expert int) bool { return expert == 0 || expert == 2 })
	if err != nil {
		t.Fatal(err)
	}
	split := ForwardScratch{MoeOut: make([]float32, len(normedRows))}
	if err := runGGUFCPUExpertsGroupedNoPostNorm(LayerOp{Layer: 0, Kind: OpExperts}, split, idx, normedRows, kept); err != nil {
		t.Fatal(err)
	}
	if err := runGGUFCPUExpertsGroupedNoPostNorm(LayerOp{Layer: 0, Kind: OpExperts}, split, idx, normedRows, dropped); err != nil {
		t.Fatal(err)
	}
	for i := range full.MoeOut {
		if math.Abs(float64(split.MoeOut[i]-full.MoeOut[i])) > 2e-3 {
			t.Fatalf("split out[%d]=%g full %g", i, split.MoeOut[i], full.MoeOut[i])
		}
	}
}
