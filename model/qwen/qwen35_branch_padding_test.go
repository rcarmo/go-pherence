package qwen

import (
	"fmt"
	"math"
	"runtime"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/tensor"
)

func TestSIMDBranchPaddingGeometry(t *testing.T) {
	block := 4
	if runtime.GOARCH == "amd64" {
		block = 6
	}
	if simd.SgemmNTRowBlock != block {
		t.Fatalf("row block=%d want %d", simd.SgemmNTRowBlock, block)
	}
	for rows := 1; rows <= 512; rows++ {
		padded := qwen35ProjectionPaddedRows(rows)
		if padded < rows || padded-rows >= block || padded%block != 0 || padded > (rows+11)/12*12 {
			t.Fatalf("rows=%d padded=%d", rows, padded)
		}
	}
}

func TestSIMDBranchPaddingProjectionError(t *testing.T) {
	// Overlapping internal buffers must propagate the checked-kernel error in
	// both padded and direct paths; don't return partial scratch as success.
	const in, out = 19, 16
	w := &tensor.Tensor{}
	for _, rows := range []int{1, simd.SgemmNTRowBlock} {
		padded := qwen35ProjectionPaddedRows(rows)
		s := &Qwen35SIMDBranch{packedOnly: true, packed: map[*tensor.Tensor][]float32{w: make([]float32, in*out)}, scratch: map[string][]float32{"padIn": make([]float32, padded*in), "padOut": make([]float32, padded*out)}}
		// Alias destination/input in the direct path, and the two private
		// scratch buffers in the padded path.
		x := make([]float32, rows*in)
		dst := x[:rows*out]
		if rows == 1 {
			backing := make([]float32, padded*in)
			s.scratch["padIn"], s.scratch["padOut"] = backing, backing[:padded*out]
		}
		if err := s.project(dst, x, w, rows, in, out); err == nil {
			t.Fatal("overlapping internal buffers accepted")
		}
	}
}

func TestSIMDBranchPaddingMatchesOldTwelve(t *testing.T) {
	oldAsm := simd.HasSgemmAsm
	defer func() { simd.HasSgemmAsm = oldAsm }()
	oldProcs := runtime.GOMAXPROCS(6)
	defer runtime.GOMAXPROCS(oldProcs)
	for _, asm := range []bool{false, oldAsm} {
		simd.HasSgemmAsm = asm
		for _, rows := range []int{1, 3, 4, 5, 6, 7, 8, 11, 12, 13, 16, 17, 18, 19, 41, 42, 48, 60, 126, 128, 255, 256, 257, 510, 511, 512} {
			t.Run(fmt.Sprintf("asm%t/rows%d", asm, rows), func(t *testing.T) {
				const in, out = 19, 528
				w := tensor.FromOwnedFloat32(make([]float32, in*out), []int{out, in})
				x := make([]float32, rows*in)
				seed := uint32(197)
				for _, s := range [][]float32{x, w.Data()} {
					for i := range s {
						seed = seed*1664525 + 1013904223
						s[i] = math.Float32frombits((seed & 0x807fffff) | 0x3e800000)
					}
				}
				packed, err := simd.PackSgemmNTWeights(w.Data(), out, in, in)
				if err != nil {
					t.Fatal(err)
				}
				padded := qwen35ProjectionPaddedRows(rows)
				inputStore, outputStore := make([]float32, padded*in+2), make([]float32, padded*out+2)
				inputStore[0], inputStore[len(inputStore)-1] = 17, -19
				outputStore[0], outputStore[len(outputStore)-1] = 23, -29
				s := &Qwen35SIMDBranch{packedOnly: true, packed: map[*tensor.Tensor][]float32{w: packed}, scratch: map[string][]float32{"padIn": inputStore[1 : len(inputStore)-1], "padOut": outputStore[1 : len(outputStore)-1]}}
				stop := s.startProjectionWorkers()
				defer stop()
				oldRows := (rows + 11) / 12 * 12
				oldIn, oldOut := make([]float32, oldRows*in), make([]float32, oldRows*out)
				dstStore := make([]float32, rows*out+2)
				dstStore[0], dstStore[len(dstStore)-1] = 31, -37
				dst := dstStore[1 : len(dstStore)-1]
				for iteration := 0; iteration < 2; iteration++ {
					// Reuse scratch with hostile stale values; only real rows may
					// reach downstream attention/recurrent state.
					for _, buffer := range [][]float32{s.scratch["padIn"], s.scratch["padOut"], dst} {
						for i := range buffer {
							buffer[i] = float32(math.NaN())
						}
					}
					for i := range x {
						x[i] = -x[i]
					}
					copy(oldIn, x)
					clear(oldIn[rows*in:])
					if err := s.projectRows(oldOut, oldIn, w, oldRows, in, out); err != nil {
						t.Fatal(err)
					}
					if err := s.project(dst, x, w, rows, in, out); err != nil {
						t.Fatal(err)
					}
					for i, v := range dst {
						if math.Float32bits(v) != math.Float32bits(oldOut[i]) {
							t.Fatalf("output[%d]=%08x old=%08x", i, math.Float32bits(v), math.Float32bits(oldOut[i]))
						}
					}
				}
				if inputStore[0] != 17 || inputStore[len(inputStore)-1] != -19 || outputStore[0] != 23 || outputStore[len(outputStore)-1] != -29 || dstStore[0] != 31 || dstStore[len(dstStore)-1] != -37 {
					t.Fatal("guard overwritten")
				}
				// Existing RISC-V microkernel still allocates; not a new claim.
				if runtime.GOARCH != "riscv64" || !asm {
					if n := testing.AllocsPerRun(10, func() {
						if err := s.project(dst, x, w, rows, in, out); err != nil {
							panic(err)
						}
					}); n != 0 {
						t.Fatalf("allocations=%g", n)
					}
				}
			})
		}
	}
}
