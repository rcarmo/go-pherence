//go:build riscv64

package whisper

import (
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"github.com/rcarmo/go-pherence/backends/spacemit/k3engine/aipool"
)

var (
	useA100FC1      = os.Getenv("WHISPER_A100_FC1") != ""
	useA100FC2      = os.Getenv("WHISPER_A100_FC2") != ""
	useA100FFNFused = os.Getenv("WHISPER_A100_FFN_FUSED") != ""
	a100FFNFC2Mode  = strings.ToLower(os.Getenv("WHISPER_A100_FFN_FC2_MODE"))
)

var (
	a100PoolOnce    sync.Once
	a100Pool        *aipool.AIWorkerPool
	a100WeightCache sync.Map // uintptr(&weight[0]) -> ime2.Q80x32
	a100Ns          int64
)

func getA100Pool() *aipool.AIWorkerPool {
	a100PoolOnce.Do(func() {
		n := 6
		if v := os.Getenv("WHISPER_A100_WORKERS"); v != "" {
			if x, err := strconv.Atoi(v); err == nil && x > 0 {
				n = x
			}
		}
		if n > 8 {
			n = 8
		}
		// Registered A100 workers can run outside the login cgroup, but they still
		// need Go scheduler Ps. Keep enough Ps for the normal X100 linears plus the
		// persistent A100 pool; otherwise enabling the pool slows unrelated work.
		needP := linearWorkers + n
		if needP > 16 {
			needP = 16
		}
		if runtime.GOMAXPROCS(0) < needP {
			runtime.GOMAXPROCS(needP)
		}
		// The experimental A100 FFN integration uses its own per-call activation M4
		// packing; generic activation TCM staging in AIWorkerPool has not measured as
		// a whole-pass win for this path. Default it off unless explicitly requested.
		if os.Getenv("IME2_TCM_ACT") == "" {
			os.Setenv("IME2_TCM_ACT", "0")
		}
		a100Pool = aipool.NewAIWorkerPool(n)
	})
	return a100Pool
}

func getA100Q80x32Weight(weight []float32, outDim, inDim int) ime2.Q80x32 {
	key := uintptr(unsafe.Pointer(&weight[0]))
	if v, ok := a100WeightCache.Load(key); ok {
		return v.(ime2.Q80x32)
	}
	q := ime2.PackF32ToQ80x32(outDim, inDim, weight)
	a100WeightCache.Store(key, q)
	return q
}

func a100LinearEligible(seqLen, inDim, outDim int) bool {
	if seqLen <= 0 {
		return false
	}
	if useA100FC1 && inDim == 1280 && outDim == 5120 {
		return true
	}
	if useA100FC2 && inDim == 5120 && outDim == 1280 {
		return true
	}
	return false
}

func linearForwardA100Raw(x, weight, bias []float32, seqLen, inDim, outDim int) ([]float32, bool) {
	w := getA100Q80x32Weight(weight, outDim, inDim)
	if !w.Valid {
		return nil, false
	}
	out := make([]float32, seqLen*outDim)
	t0 := nowNs()
	ok := aipool.GemmQ80x32AIPooled(x, seqLen, inDim, w, out, getA100Pool())
	a100Ns += nowNs() - t0
	if !ok {
		return nil, false
	}
	if bias != nil {
		for i := 0; i < seqLen; i++ {
			row := out[i*outDim : (i+1)*outDim]
			for j := 0; j < outDim && j < len(bias); j++ {
				row[j] += bias[j]
			}
		}
	}
	return out, true
}

func linearForwardA100FC1(x, weight, bias []float32, seqLen, inDim, outDim int) ([]float32, bool) {
	if !a100LinearEligible(seqLen, inDim, outDim) {
		return nil, false
	}
	return linearForwardA100Raw(x, weight, bias, seqLen, inDim, outDim)
}

func a100FFNUsesA100FC2() bool {
	switch a100FFNFC2Mode {
	case "", "a100":
		return true
	case "int8", "native", "x100":
		return false
	default:
		return true
	}
}

func forwardA100FFNFC1NativeFC2Raw(mlpIn []float32, layer *EncoderLayer, residual []float32, seqLen, dModel, ffnDim int) ([]float32, bool) {
	if dModel != 1280 || ffnDim != 5120 || layer == nil {
		return nil, false
	}
	hidden, ok := linearForwardA100Raw(mlpIn, layer.FC1Weight, layer.FC1Bias, seqLen, dModel, ffnDim)
	if !ok {
		return nil, false
	}
	gelu(hidden)
	out := linearForwardOpt(hidden, layer.FC2Weight, layer.FC2Bias, seqLen, ffnDim, dModel)
	for i := range residual {
		out[i] += residual[i]
	}
	return out, true
}

func forwardA100FFNFusedRaw(mlpIn []float32, layer *EncoderLayer, residual []float32, seqLen, dModel, ffnDim int) ([]float32, bool) {
	if dModel != 1280 || ffnDim != 5120 || layer == nil {
		return nil, false
	}
	hidden, ok := linearForwardA100Raw(mlpIn, layer.FC1Weight, layer.FC1Bias, seqLen, dModel, ffnDim)
	if !ok {
		return nil, false
	}
	w2 := getA100Q80x32Weight(layer.FC2Weight, dModel, ffnDim)
	if !w2.Valid {
		return nil, false
	}
	out := make([]float32, seqLen*dModel)
	t0 := nowNs()
	ok = aipool.GemmQ80x32AIPooledGELU(hidden, seqLen, ffnDim, w2, out, getA100Pool())
	a100Ns += nowNs() - t0
	if !ok {
		return nil, false
	}
	for i := 0; i < seqLen; i++ {
		row := out[i*dModel : (i+1)*dModel]
		for j := 0; j < dModel; j++ {
			if j < len(layer.FC2Bias) {
				row[j] += layer.FC2Bias[j]
			}
			row[j] += residual[i*dModel+j]
		}
	}
	return out, true
}

func forwardA100FFNTile(mlpIn []float32, layer *EncoderLayer, residual []float32, seqLen, dModel, ffnDim int) ([]float32, bool) {
	if !useA100FFNFused {
		return nil, false
	}
	if !a100FFNUsesA100FC2() {
		return forwardA100FFNFC1NativeFC2Raw(mlpIn, layer, residual, seqLen, dModel, ffnDim)
	}
	return forwardA100FFNFusedRaw(mlpIn, layer, residual, seqLen, dModel, ffnDim)
}

func forwardA100FFNFused(mlpIn []float32, layer *EncoderLayer, residual []float32, seqLen, dModel, ffnDim int) ([]float32, bool) {
	if !useA100FFNFused {
		return nil, false
	}
	if !a100FFNUsesA100FC2() {
		return forwardA100FFNFC1NativeFC2Raw(mlpIn, layer, residual, seqLen, dModel, ffnDim)
	}
	return forwardA100FFNFusedRaw(mlpIn, layer, residual, seqLen, dModel, ffnDim)
}

func prepackA100EncoderWeights(enc *Encoder) {
	if enc == nil || (!useA100FC1 && !useA100FC2 && !useA100FFNFused) {
		return
	}
	_ = getA100Pool()
	for i := range enc.Layers {
		layer := &enc.Layers[i]
		if (useA100FC1 || useA100FFNFused) && len(layer.FC1Weight) != 0 {
			_ = getA100Q80x32Weight(layer.FC1Weight, 5120, 1280)
		}
		if (useA100FC2 || (useA100FFNFused && a100FFNUsesA100FC2())) && len(layer.FC2Weight) != 0 {
			_ = getA100Q80x32Weight(layer.FC2Weight, 1280, 5120)
		}
	}
}

func resetA100Timers()       { a100Ns = 0 }
func a100TimingLine() string { return "[a100] ffn=" + time.Duration(a100Ns).String() }
