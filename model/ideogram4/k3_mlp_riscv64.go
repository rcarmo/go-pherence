//go:build riscv64

package ideogram4

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	simdruntime "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/backends/spacemit/k3engine/aipool"
	"github.com/rcarmo/go-pherence/backends/spacemit/rvv"
)

func k3FastSiLUMode() string {
	return strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_K3_FAST_SILU")))
}

func k3FastSiLUEnabled() bool {
	v := k3FastSiLUMode()
	return v == "1" || v == "true" || v == "yes" || v == "on" || v == "lut"
}

const k3SiLUTableN = 4096

var (
	k3SiLUTableOnce sync.Once
	k3SiLUTable     [k3SiLUTableN + 1]float32
)

func k3InitSiLUTable() {
	const lo, hi = float32(-8), float32(8)
	for i := 0; i <= k3SiLUTableN; i++ {
		x := lo + (hi-lo)*float32(i)/k3SiLUTableN
		k3SiLUTable[i] = siluScalar(x)
	}
}

func k3SiLULUT(x float32) float32 {
	const lo, hi = float32(-8), float32(8)
	if x <= lo {
		return 0
	}
	if x >= hi {
		return x
	}
	k3SiLUTableOnce.Do(k3InitSiLUTable)
	pos := (x - lo) * (k3SiLUTableN / (hi - lo))
	i := int(pos)
	if i < 0 {
		i = 0
	}
	if i >= k3SiLUTableN {
		i = k3SiLUTableN - 1
	}
	frac := pos - float32(i)
	return k3SiLUTable[i]*(1-frac) + k3SiLUTable[i+1]*frac
}

func k3A100MLPEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_K3_A100_MLP")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func k3SiLUTo(dst, x []float32) bool {
	if !k3Enabled() || len(dst) != len(x) || len(x) == 0 {
		return false
	}
	// K3 runtime seam for SiLU vector activation. Current body preserves scalar
	// semantics; replace with RVV approximation/table kernel when added.
	for i, v := range x {
		dst[i] = siluScalar(v)
	}
	return true
}

func k3MulTo(dst, a, b []float32) bool {
	if !k3Enabled() || len(dst) != len(a) || len(dst) != len(b) || len(dst) == 0 {
		return false
	}
	simdruntime.VecMul(dst, a, b)
	return true
}

func k3SiLUMulInPlace(gate, up []float32) bool {
	if !k3Enabled() || len(gate) != len(up) || len(gate) == 0 {
		return false
	}
	rvv.SiLUMulRVV(gate, gate, up)
	return true
}

func k3MLPBatch(l DiTLayer, x, out []float32, batch int) (bool, error) {
	if !k3Enabled() || !k3A100Q8Enabled() || !k3A100MLPEnabled() {
		return false, nil
	}
	if l.W1 == nil || l.W2 == nil || l.W3 == nil {
		return false, nil
	}
	if batch <= 0 {
		return true, fmt.Errorf("ideogram4 K3 A100 MLP invalid batch=%d", batch)
	}
	if len(x) < batch*l.W1.InDim() || len(out) < batch*l.W2.OutDim() {
		return true, fmt.Errorf("ideogram4 K3 A100 MLP invalid buffers x=%d/%d out=%d/%d", len(x), batch*l.W1.InDim(), len(out), batch*l.W2.OutDim())
	}
	if l.W1.InDim() != l.W3.InDim() || l.W1.OutDim() != l.W3.OutDim() || l.W2.InDim() != l.W1.OutDim() {
		return false, nil
	}
	inter := l.W1.OutDim()
	gAll := make([]float32, batch*inter)
	uAll := make([]float32, batch*inter)
	t0 := time.Now()
	w1 := l.W1.k3.ensureWeightQ80RowScale(l.W1)
	w3 := l.W3.k3.ensureWeightQ80RowScale(l.W3)
	if !aipool.Gemm2Q80x32AIPooledX100PackSameInput(x, batch, l.W1.InDim(), w1, w3, gAll, uAll, k3A100WorkerPool()) {
		return false, nil
	}
	timingMarkDiTSub("mlp_w1w3", t0)
	t0 = time.Now()
	if l.W1.weight.Bias != nil {
		for b := 0; b < batch; b++ {
			row := gAll[b*inter : (b+1)*inter]
			for i, bias := range l.W1.weight.Bias {
				row[i] += bias
			}
		}
	}
	if l.W3.weight.Bias != nil {
		for b := 0; b < batch; b++ {
			row := uAll[b*inter : (b+1)*inter]
			for i, bias := range l.W3.weight.Bias {
				row[i] += bias
			}
		}
	}
	siluMulInPlace(gAll, uAll)
	timingMarkDiTSub("mlp_silu_mul", t0)
	t0 = time.Now()
	if err := l.W2.ApplyBatch(gAll, out, batch); err != nil {
		return true, err
	}
	timingMarkDiTSub("mlp_w2", t0)
	return true, nil
}
