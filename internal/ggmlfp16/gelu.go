package ggmlfp16

import (
	"math"
	"sync"

	"github.com/rcarmo/go-pherence/half"
)

var geluTableOnce sync.Once
var geluTable [1 << 16]uint16

func initGELUTable() {
	const sqrt2OverPi = float32(0.79788456080286535587989211986876)
	for i := range geluTable {
		x := half.F16ToF32(uint16(i))
		inner := sqrt2OverPi * x * (1 + 0.044715*x*x)
		g := 0.5 * x * (1 + float32(math.Tanh(float64(inner))))
		geluTable[i] = half.F32ToF16(g)
	}
}

func GELUFP16Lookup(x float32) float32 {
	if x <= -10.0 {
		return 0
	}
	if x >= 10.0 {
		return x
	}
	geluTableOnce.Do(initGELUTable)
	return half.F16ToF32(geluTable[half.F32ToF16(x)])
}

func GELUFP16LookupMulTo(dst, gate, up []float32) bool {
	if len(dst) != len(gate) || len(dst) != len(up) {
		return false
	}
	for i := range dst {
		dst[i] = GELUFP16Lookup(gate[i]) * up[i]
	}
	return true
}
