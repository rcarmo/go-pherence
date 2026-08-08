package gguf

import (
	"encoding/binary"
	"fmt"
	"math"

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

func dotQ4_0x8Q8_0x4LlamaVNNI(q4, q8 []byte, blocks int, out *[32]float32) error {
	return llamaq4.DotQ4_0x8Q8_0x4VNNI(q4, q8, blocks, out)
}

// projectQ4_0LlamaExperimental is a non-production entry point consuming only
// prepacked panels. The retained projection dispatcher never calls it.
func projectQ4_0LlamaExperimental(q4, q8 []byte, rows, tokens, blocks int, out []float32) error {
	return llamaq4.ProjectQ4_0x8Q8_0x4VNNI(q4, q8, rows, tokens, blocks, out)
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
