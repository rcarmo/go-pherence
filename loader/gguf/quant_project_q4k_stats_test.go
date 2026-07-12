package gguf

import (
	"encoding/binary"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestProjectBatchQ4KReusesPackedTilesAndActGroups(t *testing.T) {
	resetExperimentalQ4K8x8Stats()
	m := syntheticProjectQ4KMatrix(t, 256, 16)
	x := make([]float32, 9*256)
	for i := range x {
		x[i] = float32((i%29)-14) * 0.015625
	}
	dst := make([]float32, 9*16)
	if err := m.ProjectBatchF32To(dst, x, 9); err != nil {
		t.Fatal(err)
	}
	stats := snapshotExperimentalQ4K8x8Stats()
	if stats.repacks != 2 {
		t.Fatalf("repack count=%d want 2", stats.repacks)
	}
	if stats.quant4x != 2 {
		t.Fatalf("4x quant count=%d want 2", stats.quant4x)
	}
	if stats.quant8x != 1 {
		t.Fatalf("8x quant count=%d want 1", stats.quant8x)
	}
	if stats.tilePairs != 4 {
		t.Fatalf("tile pair count=%d want 4", stats.tilePairs)
	}
}

func BenchmarkProjectBatchQ4KAllocations(b *testing.B) {
	m := syntheticProjectQ4KMatrix(b, 2816, 64)
	x := make([]float32, 8*2816)
	for i := range x {
		x[i] = float32((i%31)-15) * 0.01
	}
	dst := make([]float32, 8*64)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := m.ProjectBatchF32To(dst, x, 8); err != nil {
			b.Fatal(err)
		}
	}
}

func syntheticProjectQ4KMatrix(t testing.TB, inDim, outDim int) *QuantMatrix {
	t.Helper()
	m := &QuantMatrix{Name: "synthetic.q4k", QType: QuantQ4_K, InDim: inDim, OutDim: outDim}
	rowBytes, err := m.RowBytes()
	if err != nil {
		t.Fatal(err)
	}
	m.Raw = make([]byte, rowBytes*outDim)
	for r := 0; r < outDim; r++ {
		row := m.Raw[r*rowBytes : (r+1)*rowBytes]
		for b := 0; b < inDim/256; b++ {
			blk := row[b*144 : (b+1)*144]
			fillSyntheticQ4KBlock(blk, r, b)
		}
	}
	return m
}

func fillSyntheticQ4KBlock(blk []byte, row, block int) {
	binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(0.025+float32(row+block)*0.002))
	binary.LittleEndian.PutUint16(blk[2:4], half.F32ToF16(0.004+float32((row+block)%3)*0.001))
	for i := 0; i < 12; i++ {
		blk[4+i] = byte(1 + (i+row+block)%17)
	}
	for i := 0; i < 128; i++ {
		blk[16+i] = byte((i*7 + row*11 + block*13) & 0xff)
	}
}
