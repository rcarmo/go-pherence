//go:build amd64

package llamaq4plan9

import (
	"fmt"
	"github.com/rcarmo/go-pherence/loader/gguf/internal/q4layout"
	"unsafe"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"golang.org/x/sys/cpu"
)

// Available reports whether the Plan 9 fused kernels can execute on this CPU.
func Available() bool {
	return cpu.X86.HasAVX2 && cpu.X86.HasAVXVNNI && cpu.X86.HasFMA && simd.RuntimeCapabilities().HasF16C
}

//go:noescape
func _go_llama_q4_0_q8_0_8x4(q4 *byte, q8 *byte, blocks int, out *float32)

//go:noescape
func _go_llama_q4_0_q8_0_8x16(q4 *byte, q8 *byte, blocks int, out *float32)

//go:noescape
func _go_llama_q4_0_q8_0_8x8(q4 *byte, q8 *byte, blocks int, out *float32)

//go:noescape
func _go_llama_q4_0_q8_0_8x8_stage(q4 *byte, q8 *byte, blocks int, out *float32)

//go:noescape
func _go_llama_q4_0_q8_0_8x8_direct(q4 *byte, q8 *byte, blocks int, out *float32, stride int)

//go:noescape
func _go_llama_q4_0_q8_0_8x16_pair_direct(q4 *byte, q8 *byte, blocks int, out *float32, stride int, constants *byte)

var stagedConstants = [64]uint32{
	0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f,
	0x88888888, 0x88888888, 0x88888888, 0x88888888, 0x88888888, 0x88888888, 0x88888888, 0x88888888,
	0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x01010101,
	4, 5, 6, 7, 0, 1, 2, 3,
	0, 0, 0, 0, 0, 0, 0, 0,
	1, 1, 1, 1, 1, 1, 1, 1,
	4, 4, 4, 4, 4, 4, 4, 4,
	5, 5, 5, 5, 5, 5, 5, 5,
}

// DotQ4_0x8Q8_0x4CompilerPlan9 is the direct Plan 9 translation of the
// retained compiler kernel. It exists as a performance baseline for the
// hand-scheduled implementation.
func DotQ4_0x8Q8_0x4CompilerPlan9(q4, q8 []byte, blocks int, out *[32]float32) error {
	if !Available() {
		return fmt.Errorf("llama Q4_0x8 kernel requires AVX2, AVX-VNNI, FMA and F16C")
	}
	if out == nil || !q4layout.Tile(len(q4), len(q8), blocks, 1) {
		return fmt.Errorf("llama tile size: q4=%d q8=%d blocks=%d", len(q4), len(q8), blocks)
	}
	_go_llama_q4_0_q8_0_8x4(unsafe.SliceData(q4), unsafe.SliceData(q8), blocks, &out[0])
	return nil
}

// DotQ4_0x8Q8_0x16CompilerPlan9 computes one fused 8-row by 16-token tile.
func DotQ4_0x8Q8_0x16CompilerPlan9(q4, q8 []byte, blocks int, out *[128]float32) error {
	if !Available() {
		return fmt.Errorf("llama Q4_0x8 kernel requires AVX2, AVX-VNNI, FMA and F16C")
	}
	if out == nil || !q4layout.Tile(len(q4), len(q8), blocks, 4) {
		return fmt.Errorf("llama 8x16 tile size: q4=%d q8=%d blocks=%d", len(q4), len(q8), blocks)
	}
	_go_llama_q4_0_q8_0_8x16(unsafe.SliceData(q4), unsafe.SliceData(q8), blocks, &out[0])
	return nil
}

// DotQ4_0x8Q8_0x8CompilerPlan9 computes one dual-panel 8-row by 8-token tile.
func DotQ4_0x8Q8_0x8CompilerPlan9(q4, q8 []byte, blocks int, out *[64]float32) error {
	if !Available() {
		return fmt.Errorf("llama Q4_0x8 kernel requires AVX2, AVX-VNNI, FMA and F16C")
	}
	if out == nil || !q4layout.Tile(len(q4), len(q8), blocks, 2) {
		return fmt.Errorf("llama 8x8 tile size: q4=%d q8=%d blocks=%d", len(q4), len(q8), blocks)
	}
	_go_llama_q4_0_q8_0_8x8(unsafe.SliceData(q4), unsafe.SliceData(q8), blocks, &out[0])
	return nil
}

// DotQ4_0x8Q8_0x16PairPlan9 computes one four-panel tile as two inlined staged
// pairs and writes token-major contiguous output.
func DotQ4_0x8Q8_0x16PairPlan9(q4, q8 []byte, blocks int, out *[128]float32) error {
	if !Available() {
		return fmt.Errorf("llama Q4_0x8 kernel requires AVX2, AVX-VNNI, FMA and F16C")
	}
	if out == nil || !q4layout.Tile(len(q4), len(q8), blocks, 4) {
		return fmt.Errorf("llama paired 8x16 tile size: q4=%d q8=%d blocks=%d", len(q4), len(q8), blocks)
	}
	_go_llama_q4_0_q8_0_8x16_pair_direct(unsafe.SliceData(q4), unsafe.SliceData(q8), blocks, &out[0], 8, (*byte)(unsafe.Pointer(&stagedConstants[0])))
	return nil
}

// DotQ4_0x8Q8_0x8StagePlan9 computes one dual-panel tile with stage-local decoded weights.
func DotQ4_0x8Q8_0x8StagePlan9(q4, q8 []byte, blocks int, out *[64]float32) error {
	if !Available() {
		return fmt.Errorf("llama Q4_0x8 kernel requires AVX2, AVX-VNNI, FMA and F16C")
	}
	if out == nil || !q4layout.Tile(len(q4), len(q8), blocks, 2) {
		return fmt.Errorf("llama staged 8x8 tile size: q4=%d q8=%d blocks=%d", len(q4), len(q8), blocks)
	}
	_go_llama_q4_0_q8_0_8x8_stage(unsafe.SliceData(q4), unsafe.SliceData(q8), blocks, &out[0])
	return nil
}

// ProjectQ4_0x8Q8_0StageRows computes aligned row groups without cgo. Full
// eight-token groups use the staged dual-panel kernel; the final four-token
// panel uses the retained-order translated tail kernel. Padded lanes are never
// copied into the logical output.
func ProjectQ4_0x8Q8_0StageRows(q4, q8 []byte, rowBase, rowGroups, rows, tokens, blocks int, out []float32) error {
	if !Available() {
		return fmt.Errorf("llama Q4_0x8 kernel requires AVX2, AVX-VNNI, FMA and F16C")
	}
	if !q4layout.Rows(len(q4), len(q8), len(out), rowBase, rowGroups, rows, tokens, blocks) {
		return fmt.Errorf("llama Plan 9 projection rows size: q4=%d q8=%d out=%d base=%d groups=%d rows=%d tokens=%d blocks=%d", len(q4), len(q8), len(out), rowBase, rowGroups, rows, tokens, blocks)
	}
	panelBytes := blocks * 136
	rowGroupBytes := blocks * 144
	for rg := 0; rg < rowGroups; rg++ {
		w := unsafe.SliceData(q4[rg*rowGroupBytes:])
		outputRow := rowBase + rg*8
		logicalRows := 8
		if remain := rows - outputRow; remain < logicalRows {
			logicalRows = remain
		}
		token := 0
		if logicalRows == 8 {
			for ; token+16 <= tokens; token += 16 {
				_go_llama_q4_0_q8_0_8x16_pair_direct(w, unsafe.SliceData(q8[(token/4)*panelBytes:]), blocks, &out[token*rows+outputRow], rows, (*byte)(unsafe.Pointer(&stagedConstants[0])))
			}
		}
		for ; token+8 <= tokens; token += 8 {
			if logicalRows == 8 {
				_go_llama_q4_0_q8_0_8x8_direct(w, unsafe.SliceData(q8[(token/4)*panelBytes:]), blocks, &out[token*rows+outputRow], rows)
				continue
			}
			var tile [64]float32
			_go_llama_q4_0_q8_0_8x8_stage(w, unsafe.SliceData(q8[(token/4)*panelBytes:]), blocks, &tile[0])
			for t := 0; t < 8; t++ {
				copy(out[(token+t)*rows+outputRow:(token+t)*rows+outputRow+logicalRows], tile[t*8:t*8+logicalRows])
			}
		}
		for ; token < tokens; token += 4 {
			var tile [32]float32
			_go_llama_q4_0_q8_0_8x4(w, unsafe.SliceData(q8[(token/4)*panelBytes:]), blocks, &tile[0])
			logicalTokens := 4
			if remain := tokens - token; remain < logicalTokens {
				logicalTokens = remain
			}
			for t := 0; t < logicalTokens; t++ {
				copy(out[(token+t)*rows+outputRow:(token+t)*rows+outputRow+logicalRows], tile[t*8:t*8+logicalRows])
			}
		}
	}
	return nil
}

// ProjectQ4_0x8Q8_0Stage computes a complete projection serially. Callers that
// parallelise row groups should use ProjectQ4_0x8Q8_0StageRows directly.
func ProjectQ4_0x8Q8_0Stage(q4, q8 []byte, rows, tokens, blocks int, out []float32) error {
	return ProjectQ4_0x8Q8_0StageRows(q4, q8, 0, q4layout.Groups(rows, 8), rows, tokens, blocks, out)
}
