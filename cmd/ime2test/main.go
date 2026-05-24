// Package main implements a minimal pure Go inference test
// that loads a GGUF model and runs one decode step using IME2.
package main

import (
	"fmt"
	"math"
	"os"
	"time"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"github.com/rcarmo/go-pherence/backends/spacemit/inference"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func main() {
	model := os.Args[1]

	t0 := time.Now()
	g, err := gguf.Open(model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open: %v\n", err)
		os.Exit(1)
	}

	// Read hyperparameters
	arch := func() string { v, _ := g.MetaString("general.architecture"); return v }()
	nEmbd := func() int { v, _ := g.MetaUint32(arch + ".embedding_length"); return int(v) }()
	nHeads := func() int { v, _ := g.MetaUint32(arch + ".attention.head_count"); return int(v) }()
	nLayers := func() int { v, _ := g.MetaUint32(arch + ".block_count"); return int(v) }()
	nVocab := func() int { v, _ := g.MetaUint32("tokenizer.ggml.vocab_size"); return int(v) }()
	fmt.Fprintf(os.Stderr, "Model: %s arch=%s embd=%d heads=%d layers=%d vocab=%d\n",
		model, arch, nEmbd, nHeads, nLayers, nVocab)

	// Load token embedding (for get_rows)
	tokEmbdT, ok := g.TensorByName("token_embd.weight")
	if !ok {
		fmt.Fprintf(os.Stderr, "token_embd.weight not found\n")
		os.Exit(1)
	}
	tokEmbdData := func() []byte { d, _ := g.Raw(tokEmbdT); return d }()
	fmt.Fprintf(os.Stderr, "token_embd: shape=%v type=%d bytes=%d\n",
		tokEmbdT.Shape, tokEmbdT.QType, len(tokEmbdData))

	// Load first layer attn_norm + wq to test the matmul pipeline
	normT, _ := g.TensorByName("blk.0.attn_norm.weight")
	normData := func() []byte { d, _ := g.Raw(normT); return d }()
	fmt.Fprintf(os.Stderr, "attn_norm.0: shape=%v bytes=%d\n", normT.Shape, len(normData))

	var wqName string
	if arch == "qwen3" {
		wqName = "blk.0.attn_q.weight"
	} else {
		wqName = "blk.0.attn_q.weight"
	}
	wqT, _ := g.TensorByName(wqName)
	wqData := func() []byte { d, _ := g.Raw(wqT); return d }()
	wqRows := int(wqT.Shape[1]) // output dim
	wqCols := int(wqT.Shape[0]) // input dim (= nEmbd)
	fmt.Fprintf(os.Stderr, "wq.0: shape=%v type=%d rows=%d cols=%d bytes=%d\n",
		wqT.Shape, wqT.QType, wqRows, wqCols, len(wqData))

	loadTime := time.Since(t0)
	fmt.Fprintf(os.Stderr, "Load: %.3fs\n\n", loadTime.Seconds())

	// Step 1: Get embedding for token 0 (BOS or test)
	// For Q6_K embeddings, need dequant. For now just test with F32.
	// Actually let's test with a synthetic F32 activation vector.
	x := make([]float32, nEmbd)
	for i := range x {
		x[i] = 0.01 * float32(i%100-50) // synthetic
	}

	// Step 2: RMS norm
	normWeights := make([]float32, nEmbd)
	// normData is F32
	for i := 0; i < nEmbd; i++ {
		bits := *(*uint32)(unsafe.Pointer(&normData[i*4]))
		normWeights[i] = math.Float32frombits(bits)
	}
	xn := make([]float32, nEmbd)
	inference.RMSNorm(x, normWeights, xn, 1e-5)

	// Step 3: Quantize wq weights to INT8 and pre-pack
	// For Q4_K, need to dequant first, then re-quantize to INT8
	// This is the naive approach — the optimized version would work directly on Q4_K
	fmt.Fprintf(os.Stderr, "Dequantizing wq (%d×%d)...\n", wqRows, wqCols)
	wqF32 := dequantQ4K(wqData, wqRows, wqCols)
	wqI8 := make([]int8, wqRows*wqCols)
	wqScale := inference.QuantizeF32ToINT8(wqF32, wqI8)
	fmt.Fprintf(os.Stderr, "Packing tiles...\n")

	// Pad dimensions to multiples of 4/8
	wqRowsPad := ((wqRows + 3) / 4) * 4
	wqColsPad := ((wqCols + 7) / 8) * 8
	wqI8Pad := make([]int8, wqRowsPad*wqColsPad)
	for r := 0; r < wqRows; r++ {
		copy(wqI8Pad[r*wqColsPad:r*wqColsPad+wqCols], wqI8[r*wqCols:(r+1)*wqCols])
	}
	wqPacked := ime2.PackTiles(wqI8Pad, wqRowsPad, wqColsPad)

	// Step 4: MatVec
	fmt.Fprintf(os.Stderr, "Running MatVecQ4K (%d×%d)...\n", wqRowsPad, wqColsPad)
	xnPad := make([]float32, wqColsPad)
	copy(xnPad, xn)
	out := make([]float32, wqRowsPad)

	t1 := time.Now()
	inference.MatVecQ4K(wqRowsPad, wqColsPad, wqPacked, xnPad, out, wqScale)
	elapsed := time.Since(t1)

	// Print result
	fmt.Printf("MatVec %d×%d: %.3fms\n", wqRows, wqCols, float64(elapsed.Microseconds())/1000)
	fmt.Printf("Output[0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
		out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7])

	// Compute reference (scalar F32)
	t2 := time.Now()
	ref := make([]float32, wqRows)
	for i := 0; i < wqRows; i++ {
		var sum float32
		for k := 0; k < wqCols; k++ {
			sum += wqF32[i*wqCols+k] * xn[k]
		}
		ref[i] = sum
	}
	refTime := time.Since(t2)
	fmt.Printf("Scalar ref: %.3fms\n", float64(refTime.Microseconds())/1000)
	fmt.Printf("Ref[0..7]:  %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
		ref[0], ref[1], ref[2], ref[3], ref[4], ref[5], ref[6], ref[7])

	// Error
	var maxErr float32
	for i := 0; i < wqRows; i++ {
		e := float32(math.Abs(float64(out[i] - ref[i])))
		if e > maxErr {
			maxErr = e
		}
	}
	var maxRef float32
	for _, v := range ref {
		if a := float32(math.Abs(float64(v))); a > maxRef {
			maxRef = a
		}
	}
	fmt.Printf("MaxErr: %.4f (%.2f%% relative)\n", maxErr, maxErr/maxRef*100)
	fmt.Printf("Speedup: %.1f×\n", float64(refTime)/float64(elapsed))
}

// dequantQ4K dequantizes Q4_K tensor data to F32.
func dequantQ4K(data []byte, rows, cols int) []float32 {
	blockSize := 256 // elements per Q4_K block
	bytesPerBlock := 144
	blocksPerRow := cols / blockSize
	out := make([]float32, rows*cols)

	for row := 0; row < rows; row++ {
		for blk := 0; blk < blocksPerRow; blk++ {
			offset := (row*blocksPerRow + blk) * bytesPerBlock
			b := data[offset : offset+bytesPerBlock]

			d := fp16(b[0], b[1])
			dmin := fp16(b[2], b[3])

			// Decode scales (simplified — 6-bit packed)
			var sc, mn [8]float32
			for i := 0; i < 4; i++ {
				sc[i] = float32(b[4+i] & 63)
				mn[i] = float32(b[8+i] & 63)
			}
			for i := 0; i < 4; i++ {
				sc[i+4] = float32((b[4+i]>>6) | ((b[12+i]&0xf)<<2))
				mn[i+4] = float32((b[8+i]>>6) | ((b[12+i]>>4)<<2))
			}

			// Dequant 256 elements (8 sub-blocks × 32 elements)
			base := row*cols + blk*blockSize
			for sb := 0; sb < 8; sb++ {
				scF := d * sc[sb]
				mnF := dmin * mn[sb]
				qOff := 16 + sb*16 // 16 bytes offset for scales, then 16 bytes per sub-block
				for j := 0; j < 16; j++ {
					q := b[qOff+j]
					out[base+sb*32+j] = scF*float32(q&0xf) - mnF
					out[base+sb*32+j+16] = scF*float32(q>>4) - mnF
				}
			}
		}
	}
	return out
}

func fp16(lo, hi byte) float32 {
	h := uint16(lo) | uint16(hi)<<8
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3ff
	} else if exp == 31 {
		return math.Float32frombits((sign << 31) | 0x7f800000 | (mant << 13))
	}
	exp = exp + (127 - 15)
	return math.Float32frombits((sign << 31) | (exp << 23) | (mant << 13))
}
