//go:build riscv64

package whisper

import (
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"github.com/rcarmo/go-pherence/backends/spacemit/k3engine/aipool"
)

var useA100FC1 = os.Getenv("WHISPER_A100_FC1") != ""

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
		// The FC1 A100 integration uses its own per-call M4 activation packing; the
		// generic TCM activation staging in AIWorkerPool is not yet a whole-pass win
		// for this path. Default it off unless the caller explicitly requests it.
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

func a100FC1Eligible(seqLen, inDim, outDim int) bool {
	return useA100FC1 && inDim == 1280 && outDim == 5120
}

func linearForwardA100FC1(x, weight, bias []float32, seqLen, inDim, outDim int) ([]float32, bool) {
	if !a100FC1Eligible(seqLen, inDim, outDim) {
		return nil, false
	}
	w := getA100Q80x32Weight(weight, outDim, inDim)
	if !w.Valid {
		return nil, false
	}
	Mp := (seqLen + 3) &^ 3
	xp := x
	if Mp != seqLen {
		xp = make([]float32, Mp*inDim)
		for i := 0; i < seqLen; i++ {
			copy(xp[i*inDim:(i+1)*inDim], x[i*inDim:(i+1)*inDim])
		}
	}
	outp := make([]float32, Mp*outDim)
	t0 := nowNs()
	ok := aipool.GemmQ80x32AIPooled(xp, Mp, inDim, w, outp, getA100Pool())
	a100Ns += nowNs() - t0
	if !ok {
		return nil, false
	}
	out := outp[:seqLen*outDim]
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

func resetA100Timers()       { a100Ns = 0 }
func a100TimingLine() string { return "[a100] fc1=" + time.Duration(a100Ns).String() }
