//go:build ggml && cgo && linux

package ggmlquant

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestVecDotRowsInitializesCPUTables(t *testing.T) {
	x := make([]float32, 256)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.13)) * 0.5
	}
	q8, err := QuantizeFromFloat(Q8_K, x)
	if err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, 210)
	for i := 192; i < 208; i++ {
		raw[i] = 1
	}
	binary.LittleEndian.PutUint16(raw[208:], half.F32ToF16(0.25))
	out := []float32{0}
	if err := VecDotRows(Q6_K, out, raw, 210, q8, 256, 1); err != nil {
		t.Fatal(err)
	}
	if out[0] == 0 {
		t.Fatal("Q6_K/Q8_K vecdot returned zero; ggml CPU lookup tables may not be initialized")
	}
}
