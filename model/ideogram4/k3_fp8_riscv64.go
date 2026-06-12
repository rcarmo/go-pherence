//go:build riscv64

package ideogram4

import (
	"encoding/binary"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/rcarmo/go-pherence/backends/simd/quant/fp8"
	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"github.com/rcarmo/go-pherence/backends/spacemit/k3engine/aipool"
	"github.com/rcarmo/go-pherence/backends/spacemit/rvv"
	"github.com/rcarmo/go-pherence/half"
)

func k3Enabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_K3")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func k3Threads() int {
	if s := strings.TrimSpace(os.Getenv("GO_PHERENCE_IDEOGRAM4_K3_THREADS")); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			return n
		}
	}
	return 8
}

func k3A100Q8Enabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_K3_A100_Q8")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func k3A100Workers() int {
	if s := strings.TrimSpace(os.Getenv("GO_PHERENCE_IDEOGRAM4_K3_A100_WORKERS")); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			if n > 8 {
				n = 8
			}
			return n
		}
	}
	return 6
}

var (
	k3A100PoolOnce sync.Once
	k3A100Pool     *aipool.AIWorkerPool
)

func k3A100WorkerPool() *aipool.AIWorkerPool {
	k3A100PoolOnce.Do(func() {
		n := k3A100Workers()
		needP := k3Threads() + n
		if needP > 16 {
			needP = 16
		}
		if runtime.GOMAXPROCS(0) < needP {
			runtime.GOMAXPROCS(needP)
		}
		// The Ideogram path pre-packs A activations on X100 goroutines before A100
		// dispatch, so generic AIWorkerPool activation TCM staging is not useful by
		// default. Leave explicit caller overrides intact.
		// Allow TCM activation staging for VAE and large-token A100 GEMMs.
		k3A100Pool = aipool.NewAIWorkerPool(n)
	})
	return k3A100Pool
}

type k3FP8Cache struct {
	mu           sync.Mutex
	weightF16    []uint16 // row-major [outDim,inDim]
	weightF16N32 []uint16 // packed N32 layout for GemmF16Outer32
	weightQ80    ime2.Q80x32
	outDim       int
	inDim        int
}

func (c *k3FP8Cache) release() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.weightF16 = nil
	c.weightF16N32 = nil
	c.weightQ80 = ime2.Q80x32{}
	c.outDim, c.inDim = 0, 0
}

func (c *k3FP8Cache) ensureWeightF16(f *FP8Linear) []uint16 {
	c.mu.Lock()
	defer c.mu.Unlock()
	outDim, inDim := f.weight.OutDim, f.weight.InDim
	if c.weightF16 != nil && c.outDim == outDim && c.inDim == inDim {
		return c.weightF16
	}
	B := make([]uint16, outDim*inDim)
	for r := 0; r < outDim; r++ {
		scale := f.weight.Scale[0]
		if len(f.weight.Scale) != 1 {
			scale = f.weight.Scale[r]
		}
		wb := f.weight.Weight[r*inDim : (r+1)*inDim]
		bb := B[r*inDim : (r+1)*inDim]
		for cidx := 0; cidx < inDim; cidx++ {
			bb[cidx] = half.F32ToF16(fp8.DecodeE4M3(wb[cidx]) * scale)
		}
	}
	c.weightF16 = B
	if outDim%32 == 0 {
		c.weightF16N32 = rvv.PackBF16N32(B, outDim, inDim)
	} else {
		c.weightF16N32 = nil
	}
	c.outDim, c.inDim = outDim, inDim
	return B
}

func (c *k3FP8Cache) weightN32(f *FP8Linear) []uint16 {
	c.ensureWeightF16(f)
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.weightF16N32
}

func (c *k3FP8Cache) ensureWeightQ80RowScale(f *FP8Linear) ime2.Q80x32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	outDim, inDim := f.weight.OutDim, f.weight.InDim
	if c.weightQ80.Valid && c.weightQ80.M == outDim && c.weightQ80.K == inDim {
		return c.weightQ80
	}
	if outDim%32 != 0 || inDim%32 != 0 || len(f.weight.Weight) < outDim*inDim {
		return ime2.Q80x32{M: outDim, K: inDim}
	}
	c.weightQ80 = packFP8LinearToQ80x32RowScale(f)
	c.outDim, c.inDim = outDim, inDim
	return c.weightQ80
}

func packFP8LinearToQ80x32RowScale(f *FP8Linear) ime2.Q80x32 {
	outDim, inDim := f.weight.OutDim, f.weight.InDim
	if outDim%32 != 0 || inDim%32 != 0 || len(f.weight.Weight) < outDim*inDim {
		return ime2.Q80x32{M: outDim, K: inDim}
	}
	groups, subs := outDim/32, inDim/32
	out := make([]byte, groups*subs*ime2.K3I8I8BTileBytes)
	rowScale := make([]float32, outDim)
	// Parallel row-scale scan
	nw := k3Threads()
	if nw > outDim {
		nw = outDim
	}
	var wg sync.WaitGroup
	wg.Add(nw)
	for wid := 0; wid < nw; wid++ {
		r0 := wid * outDim / nw
		r1 := (wid + 1) * outDim / nw
		go func() {
			defer wg.Done()
			for r := r0; r < r1; r++ {
				scale := f.weight.Scale[0]
				if len(f.weight.Scale) != 1 {
					scale = f.weight.Scale[r]
				}
				maxAbs := float32(0)
				base := r * inDim
				for k := 0; k < inDim; k++ {
					v := fp8.DecodeE4M3(f.weight.Weight[base+k]) * scale
					if v < 0 {
						v = -v
					}
					if v > maxAbs {
						maxAbs = v
					}
				}
				if maxAbs != 0 {
					rowScale[r] = maxAbs / 127.0
				}
			}
		}()
	}
	wg.Wait()
	wg.Add(nw)
	for wid := 0; wid < nw; wid++ {
		g0 := wid * groups / nw
		g1 := (wid + 1) * groups / nw
		go func() {
			defer wg.Done()
			for g := g0; g < g1; g++ {
				for sb := 0; sb < subs; sb++ {
					block := out[(g*subs+sb)*ime2.K3I8I8BTileBytes:]
					scales := block[:64]
					qs := block[64 : 64+1024]
					for rr := 0; rr < 32; rr++ {
						r := g*32 + rr
						inputScale := f.weight.Scale[0]
						if len(f.weight.Scale) != 1 {
							inputScale = f.weight.Scale[r]
						}
						d := rowScale[r]
						binary.LittleEndian.PutUint16(scales[rr*2:], half.F32ToF16(d))
						inv := float32(0)
						if d != 0 {
							inv = 1 / d
						}
						base := r*inDim + sb*32
						for k := 0; k < 32; k++ {
							q := int(fp8.DecodeE4M3(f.weight.Weight[base+k]) * inputScale * inv)
							// Match existing packers' round-to-nearest behavior.
							v := fp8.DecodeE4M3(f.weight.Weight[base+k]) * inputScale * inv
							if v >= 0 {
								q = int(v + 0.5)
							} else {
								q = int(v - 0.5)
							}
							if q > 127 {
								q = 127
							}
							if q < -128 {
								q = -128
							}
							qs[rr*32+k] = byte(int8(q))
						}
					}
				}
			}
		}()
	}
	wg.Wait()
	return ime2.Q80x32{M: outDim, K: inDim, BData: out, Valid: true}
}

func k3FP8Batch(f *FP8Linear, x, out []float32, batch int) (bool, error) {
	if f == nil || !k3Enabled() || batch <= 0 {
		return false, nil
	}
	inDim, outDim := f.weight.InDim, f.weight.OutDim
	if len(x) < batch*inDim || len(out) < batch*outDim {
		return true, fmt.Errorf("ideogram4 K3 FP8 linear %q invalid buffers x=%d/%d out=%d/%d", f.spec.Prefix, len(x), batch*inDim, len(out), batch*outDim)
	}
	if k3A100Q8Enabled() && inDim%32 == 0 && outDim%32 == 0 {
		wq := f.k3.ensureWeightQ80RowScale(f)
		if wq.Valid {
			if ok := aipool.GemmQ80x32AIPooledX100Pack(x[:batch*inDim], batch, inDim, wq, out[:batch*outDim], k3A100WorkerPool()); ok {
				if f.weight.Bias != nil {
					for b := 0; b < batch; b++ {
						row := out[b*outDim : (b+1)*outDim]
						for i, bias := range f.weight.Bias {
							row[i] += bias
						}
					}
				}
				return true, nil
			}
		}
	}

	// First K3 SIMD coverage path: decode FP8 weights to fp16 rows and convert
	// F32 activations to fp16, then use the existing RVV/Zvfh fp16 GEMM kernels.
	// This is intentionally conservative and correctness-oriented; later K3 work
	// can replace it with resident packed FP8→int8/IME2 kernels.
	A := make([]uint16, batch*inDim)
	rvv.F32ToF16RVV(A, x[:batch*inDim])
	if batch%4 == 0 && outDim%32 == 0 {
		if bp := f.k3.weightN32(f); len(bp) == outDim*inDim {
			rvv.GemmF16Outer32(A, bp, out[:batch*outDim], batch, outDim, inDim, k3Threads())
		} else {
			B := f.k3.ensureWeightF16(f)
			rvv.GemmF16Threaded(A, B, out[:batch*outDim], batch, outDim, inDim, k3Threads())
		}
	} else {
		B := f.k3.ensureWeightF16(f)
		rvv.GemmF16Threaded(A, B, out[:batch*outDim], batch, outDim, inDim, k3Threads())
	}
	if f.weight.Bias != nil {
		for b := 0; b < batch; b++ {
			row := out[b*outDim : (b+1)*outDim]
			for i, bias := range f.weight.Bias {
				row[i] += bias
			}
		}
	}
	return true, nil
}
