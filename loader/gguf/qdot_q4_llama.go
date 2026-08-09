package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf/llamaq4"
)

// These layouts and the arithmetic below intentionally follow llama.cpp b607
// (065d9d50152486590c09b31627ecaf76ceba39dd), rather than Go's retained
// eight-lane FP32 accumulation order. They remain experimental.
const (
	q4_0x8BlockBytes = 144 // eight fp16 scales, then 8*32/2 packed quants
	q8_0x4BlockBytes = 136 // four fp16 scales, then 4*32 interleaved quants
)

// packQ4_0x8 packs row-major GGML Q4_0 blocks in groups of eight rows. Missing
// rows in the final group have zero scale, defining a harmless row tail.
func packQ4_0x8(raw []byte, rows, blocks int) ([]byte, error) {
	groups := (rows + 7) / 8
	if rows < 0 || groups < 0 || blocks < 0 || groups > 0 && blocks > int(^uint(0)>>1)/groups/q4_0x8BlockBytes {
		return nil, fmt.Errorf("Q4_0x8 packed size overflows")
	}
	out := make([]byte, groups*blocks*q4_0x8BlockBytes)
	if err := packQ4_0x8To(out, raw, rows, blocks); err != nil {
		return nil, err
	}
	return out, nil
}

func packQ4_0x8To(out, raw []byte, rows, blocks int) error {
	if rows < 0 || blocks < 0 || rows > 0 && blocks > int(^uint(0)>>1)/rows/18 || len(raw) != rows*blocks*18 {
		return fmt.Errorf("Q4_0x8 pack size: raw=%d rows=%d blocks=%d", len(raw), rows, blocks)
	}
	groups := (rows + 7) / 8
	if groups > 0 && blocks > int(^uint(0)>>1)/groups/q4_0x8BlockBytes || len(out) != groups*blocks*q4_0x8BlockBytes {
		return fmt.Errorf("Q4_0x8 destination size: dst=%d rows=%d blocks=%d", len(out), rows, blocks)
	}
	clear(out)
	rowBytes := blocks * 18
	for group := 0; group < groups; group++ {
		for block := 0; block < blocks; block++ {
			dst := out[(group*blocks+block)*q4_0x8BlockBytes:]
			for row := 0; row < 8; row++ {
				srcRow := group*8 + row
				if srcRow >= rows {
					continue
				}
				src := raw[srcRow*rowBytes+block*18:]
				copy(dst[row*2:], src[:2])
				for chunk := 0; chunk < 2; chunk++ {
					for j := 0; j < 8; j++ {
						dst[16+(chunk*8+row)*8+j] = src[2+chunk*8+j] ^ 0x88
					}
				}
			}
		}
	}
	return nil
}

// PrepareLlamaQ4_0x8 replaces canonical Q4_0 bytes with the size-preserving
// fused layout when this build and CPU can execute it. Callers must select only
// projection matrices whose direct Raw bytes are not handed to another backend.
// UsesLlamaQ4_0x8 reports whether canonical Raw storage was replaced by the
// fused Q4_0 layout.
func (m *QuantMatrix) UsesLlamaQ4_0x8() bool {
	return m != nil && len(m.llamaQ4_0x8) > 0
}

func (m *QuantMatrix) PrepareLlamaQ4_0x8() (bool, error) {
	if m == nil {
		return false, fmt.Errorf("nil quant matrix")
	}
	if m.QType != QuantQ4_0 || !llamaq4.Available() {
		return false, nil
	}
	if len(m.llamaQ4_0x8) > 0 {
		return true, nil
	}
	if m.InDim <= 0 || m.InDim%qk8_0 != 0 || m.OutDim <= 0 {
		return false, fmt.Errorf("quant matrix %s: invalid Q4_0x8 dimensions %dx%d", m.Name, m.OutDim, m.InDim)
	}
	packed, err := packQ4_0x8(m.Raw, m.OutDim, m.InDim/qk8_0)
	if err != nil {
		return false, fmt.Errorf("quant matrix %s: %w", m.Name, err)
	}
	m.llamaQ4_0x8 = packed
	m.Raw = nil
	return true, nil
}

func unpackQ4_0x8RowTo(dst, packed []byte, row, rows, blocks int) error {
	groups := (rows + 7) / 8
	if row < 0 || row >= rows || blocks <= 0 || len(dst) != blocks*18 || len(packed) != groups*blocks*q4_0x8BlockBytes {
		return fmt.Errorf("Q4_0x8 unpack size: dst=%d packed=%d row=%d rows=%d blocks=%d", len(dst), len(packed), row, rows, blocks)
	}
	group, lane := row/8, row%8
	for block := 0; block < blocks; block++ {
		src := packed[(group*blocks+block)*q4_0x8BlockBytes:]
		out := dst[block*18:]
		copy(out[:2], src[lane*2:lane*2+2])
		for chunk := 0; chunk < 2; chunk++ {
			for j := 0; j < 8; j++ {
				out[2+chunk*8+j] = src[16+(chunk*8+lane)*8+j] ^ 0x88
			}
		}
	}
	return nil
}

// packQ8_0x4 packs row-major Q8_0 activation blocks in groups of four tokens.
// FP32 scales are rounded to FP16 exactly as llama.cpp's block_q8_0x4. Missing
// tokens in the final group are zero-filled, defining the token tail.
func packQ8_0x4(y []q8_0Block, tokens, blocks int) ([]byte, error) {
	groups := (tokens + 3) / 4
	if tokens < 0 || groups < 0 || blocks < 0 || groups > 0 && blocks > int(^uint(0)>>1)/groups/q8_0x4BlockBytes {
		return nil, fmt.Errorf("Q8_0x4 packed size overflows")
	}
	out := make([]byte, groups*blocks*q8_0x4BlockBytes)
	if err := packQ8_0x4To(out, y, tokens, blocks); err != nil {
		return nil, err
	}
	return out, nil
}

func packQ8_0x4To(out []byte, y []q8_0Block, tokens, blocks int) error {
	if tokens < 0 || blocks < 0 || tokens > 0 && blocks > int(^uint(0)>>1)/tokens || len(y) != tokens*blocks {
		return fmt.Errorf("Q8_0x4 pack size: blocks=%d tokens=%d width-blocks=%d", len(y), tokens, blocks)
	}
	groups := (tokens + 3) / 4
	if groups > 0 && blocks > int(^uint(0)>>1)/groups/q8_0x4BlockBytes || len(out) != groups*blocks*q8_0x4BlockBytes {
		return fmt.Errorf("Q8_0x4 destination size: dst=%d tokens=%d blocks=%d", len(out), tokens, blocks)
	}
	clear(out)
	for group := 0; group < groups; group++ {
		for block := 0; block < blocks; block++ {
			dst := out[(group*blocks+block)*q8_0x4BlockBytes:]
			for token := 0; token < 4; token++ {
				srcToken := group*4 + token
				if srcToken >= tokens {
					continue
				}
				src := &y[srcToken*blocks+block]
				binary.LittleEndian.PutUint16(dst[token*2:], half.F32ToF16(src.d))
				for chunk := 0; chunk < 4; chunk++ {
					for j := 0; j < 8; j++ {
						dst[8+(chunk*4+token)*8+j] = byte(src.qs[chunk*8+j])
					}
				}
			}
		}
	}
	return nil
}

// quantizeQ8_0x4To writes the fused activation layout directly from row-major
// F32 tokens. It is byte-identical to quantizeQ8_0To followed by packQ8_0x4To,
// but does not allocate or traverse an intermediate token-major Q8 slice.
func quantizeQ8_0x4To(out []byte, x []float32, tokens, width int) error {
	if tokens < 0 || width <= 0 || width%qk8_0 != 0 || tokens > 0 && width > int(^uint(0)>>1)/tokens || len(x) != tokens*width {
		return fmt.Errorf("Q8_0x4 quantize size: len=%d tokens=%d width=%d", len(x), tokens, width)
	}
	blocks := width / qk8_0
	groups := (tokens + 3) / 4
	if groups > 0 && blocks > int(^uint(0)>>1)/groups/q8_0x4BlockBytes || len(out) != groups*blocks*q8_0x4BlockBytes {
		return fmt.Errorf("Q8_0x4 quantize destination: dst=%d tokens=%d width=%d", len(out), tokens, width)
	}
	clear(out)
	quantizeGroups := func(start, end int) {
		for group := start; group < end; group++ {
			for block := 0; block < blocks; block++ {
				dst := out[(group*blocks+block)*q8_0x4BlockBytes:]
				for token := 0; token < 4; token++ {
					srcToken := group*4 + token
					if srcToken >= tokens {
						continue
					}
					row := x[srcToken*width+block*qk8_0 : srcToken*width+(block+1)*qk8_0]
					var amax float32
					for _, v := range row {
						av := float32(math.Abs(float64(v)))
						if av > amax {
							amax = av
						}
					}
					d := amax / 127.0
					id := float32(0)
					if d != 0 {
						id = 1 / d
					}
					binary.LittleEndian.PutUint16(dst[token*2:], half.F32ToF16(d))
					for j, v := range row {
						q := int(math.Round(float64(v * id)))
						if q > 127 {
							q = 127
						}
						if q < -128 {
							q = -128
						}
						chunk, offset := j/8, j&7
						dst[8+(chunk*4+token)*8+offset] = byte(int8(q))
					}
				}
			}
		}
	}
	workers := runtime.GOMAXPROCS(0)
	if tokens < 64 || workers < 2 || groups < 2 {
		quantizeGroups(0, groups)
		return nil
	}
	if workers > groups {
		workers = groups
	}
	chunkGroups := (groups + workers*4 - 1) / (workers * 4)
	var next atomic.Int64
	work := func() {
		for {
			start := int(next.Add(int64(chunkGroups))) - chunkGroups
			if start >= groups {
				return
			}
			end := start + chunkGroups
			if end > groups {
				end = groups
			}
			quantizeGroups(start, end)
		}
	}
	var wg sync.WaitGroup
	wg.Add(workers - 1)
	for worker := 1; worker < workers; worker++ {
		go func() {
			defer wg.Done()
			work()
		}()
	}
	work()
	wg.Wait()
	return nil
}

func dotQ4_0x8Q8_0x4LlamaVNNI(q4, q8 []byte, blocks int, out *[32]float32) error {
	return llamaq4.DotQ4_0x8Q8_0x4VNNI(q4, q8, blocks, out)
}

// projectQ4_0LlamaExperimental is a non-production entry point consuming only
// prepacked panels. Full groups use one fused 8x16 kernel; the retained
// projection dispatcher never calls it.
func projectBatchQ4_0LlamaExperimental(q4 []byte, out, x []float32, rows, tokens, width int) error {
	if width <= 0 || width%qk8_0 != 0 || len(x) != tokens*width {
		return fmt.Errorf("llama F32 projection size: x=%d rows=%d tokens=%d width=%d", len(x), rows, tokens, width)
	}
	blocks := width / qk8_0
	q8 := make([]byte, (tokens+3)/4*blocks*q8_0x4BlockBytes)
	if err := quantizeQ8_0x4To(q8, x, tokens, width); err != nil {
		return err
	}
	return projectQ4_0LlamaExperimental(q4, q8, rows, tokens, blocks, out)
}

func projectQ4_0LlamaExperimental(q4, q8 []byte, rows, tokens, blocks int, out []float32) error {
	rowGroups, tokenGroups := (rows+7)/8, (tokens+3)/4
	if rows <= 0 || tokens <= 0 || blocks <= 0 || len(q4) != rowGroups*blocks*q4_0x8BlockBytes || len(q8) != tokenGroups*blocks*q8_0x4BlockBytes || len(out) != rows*tokens {
		return fmt.Errorf("llama projection size: q4=%d q8=%d out=%d rows=%d tokens=%d blocks=%d", len(q4), len(q8), len(out), rows, tokens, blocks)
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 2 || rowGroups < 2 {
		return llamaq4.ProjectQ4_0x8Q8_0x4VNNI(q4, q8, rows, tokens, blocks, out)
	}
	if workers > rowGroups {
		workers = rowGroups
	}
	chunkGroups := (rowGroups + workers*4 - 1) / (workers * 4)
	var next atomic.Int64
	var failed atomic.Bool
	var once sync.Once
	var projectErr error
	work := func() {
		for !failed.Load() {
			start := int(next.Add(int64(chunkGroups))) - chunkGroups
			if start >= rowGroups {
				return
			}
			end := start + chunkGroups
			if end > rowGroups {
				end = rowGroups
			}
			first := start * blocks * q4_0x8BlockBytes
			last := end * blocks * q4_0x8BlockBytes
			if err := llamaq4.ProjectQ4_0x8Q8_0x4RowsVNNI(q4[first:last], q8, start*8, end-start, rows, tokens, blocks, out); err != nil {
				once.Do(func() { projectErr = err })
				failed.Store(true)
				return
			}
		}
	}
	var wg sync.WaitGroup
	wg.Add(workers - 1)
	for worker := 1; worker < workers; worker++ {
		go func() {
			defer wg.Done()
			work()
		}()
	}
	work()
	wg.Wait()
	return projectErr
}

// dotQ4_0x8Q8_0x4LlamaReference computes one 8-row by 4-token tile. Each
// 32-element integer dot completes before one FP32 FMA per output/block.
func dotQ4_0x8Q8_0x4LlamaReference(q4, q8 []byte, blocks int, out *[32]float32) error {
	if blocks < 0 || len(q4) != blocks*q4_0x8BlockBytes || len(q8) != blocks*q8_0x4BlockBytes {
		return fmt.Errorf("llama tile size: q4=%d q8=%d blocks=%d", len(q4), len(q8), blocks)
	}
	*out = [32]float32{}
	for block := 0; block < blocks; block++ {
		w := q4[block*q4_0x8BlockBytes:]
		a := q8[block*q8_0x4BlockBytes:]
		for token := 0; token < 4; token++ {
			ad := half.F16ToF32(binary.LittleEndian.Uint16(a[token*2:]))
			for row := 0; row < 8; row++ {
				var dot int32
				for k := 0; k < 32; k++ {
					byteIndex := k & 15
					packed := w[16+((byteIndex/8)*8+row)*8+(byteIndex&7)]
					var nibble byte
					if k < 16 {
						nibble = packed & 0x0f
					} else {
						nibble = packed >> 4
					}
					q4v := int8(nibble<<4) >> 4
					q8v := int8(a[8+((k/8)*4+token)*8+(k&7)])
					dot += int32(q4v) * int32(q8v)
				}
				wd := half.F16ToF32(binary.LittleEndian.Uint16(w[row*2:]))
				idx := token*8 + row
				out[idx] = float32(math.FMA(float64(dot), float64(wd*ad), float64(out[idx])))
			}
		}
	}
	return nil
}
