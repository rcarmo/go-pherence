package model

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/runtime/expertstream"
)

func TestMLXWeightFromStreamComponentExactViews(t *testing.T) {
	q := &expertstream.QuantSpec{OutDim: 4, InDim: 8, GroupSize: 4, Bits: 4}
	raw := make([]byte, 80)
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint32(raw[i*4:], uint32(10+i))
	}
	for i := 0; i < 8; i++ {
		binary.LittleEndian.PutUint32(raw[16+i*4:], math.Float32bits(float32(i)+0.25))
		binary.LittleEndian.PutUint32(raw[48+i*4:], math.Float32bits(float32(i)+0.5))
	}
	got, err := mlxWeightFromStreamComponent(expertstream.Component{DType: expertstream.DTypeMLXQuant, Bytes: raw, Quant: q})
	if err != nil {
		t.Fatal(err)
	}
	if got.InDim != 8 || got.OutDim != 4 || got.GroupSize != 4 || got.Groups != 2 || got.Bits != 4 {
		t.Fatalf("metadata=%+v", got)
	}
	if got.Weight[3] != 13 || got.Scales[7] != 7.25 || got.Biases[7] != 7.5 {
		t.Fatalf("views weight=%v scales=%v biases=%v", got.Weight, got.Scales, got.Biases)
	}
	// Prove zero-copy aliasing.
	binary.LittleEndian.PutUint32(raw[:4], 99)
	if got.Weight[0] != 99 {
		t.Fatalf("weight view copied: got %d", got.Weight[0])
	}
}

func TestMLXWeightFromStreamComponentRejectsMalformed(t *testing.T) {
	if _, err := mlxWeightFromStreamComponent(expertstream.Component{}); err == nil {
		t.Fatal("expected missing quant error")
	}
	q := &expertstream.QuantSpec{OutDim: 4, InDim: 8, GroupSize: 4, Bits: 8}
	if _, err := mlxWeightFromStreamComponent(expertstream.Component{DType: expertstream.DTypeMLXQuant, Bytes: make([]byte, 72), Quant: q}); err == nil {
		t.Fatal("expected unsupported bits error")
	}
	q.Bits = 4
	if _, err := mlxWeightFromStreamComponent(expertstream.Component{DType: expertstream.DTypeMLXQuant, Bytes: make([]byte, 79), Quant: q}); err == nil {
		t.Fatal("expected size error")
	}
}
