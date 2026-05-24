// cmd/ime2run/main.go — Pure Go inference using IME2 vmadot.
// Loads a GGUF model and runs greedy decode without any CGo.
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"time"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"github.com/rcarmo/go-pherence/backends/spacemit/inference"
	"github.com/rcarmo/go-pherence/loader/gguf"
	// tokenizer loaded via gguf
)


// extractQ4KDirect extracts Q4K 4-bit nibbles to INT8 without F32 intermediate.
// Values are in range 0-15 (unsigned, stored as signed int8).

// extractQ4KScales extracts per-sub-block scales and mins from Q4K raw data.
func extractQ4KScales(data []byte, rows, cols int) ([]float32, []float32) {
	bytesPerBlock := 144
	blocksPerRow := cols / 256
	n := rows * blocksPerRow * 8
	scales := make([]float32, n)
	mins := make([]float32, n)
	for row := 0; row < rows; row++ {
		for blk := 0; blk < blocksPerRow; blk++ {
			offset := (row*blocksPerRow + blk) * bytesPerBlock
			b := data[offset : offset+bytesPerBlock]
			d := fp16ToFloat(uint16(b[0]) | uint16(b[1])<<8)
			dmin := fp16ToFloat(uint16(b[2]) | uint16(b[3])<<8)
			var sc, mn [8]float32
			for i := 0; i < 4; i++ {
				sc[i] = float32(b[4+i] & 63)
				mn[i] = float32(b[8+i] & 63)
			}
			for i := 0; i < 4; i++ {
				sc[i+4] = float32((b[12+i]&0xF) | (uint8(b[4+i]>>6)<<4))
				mn[i+4] = float32((b[12+i]>>4) | (uint8(b[8+i]>>6)<<4))
			}
			idx := (row*blocksPerRow + blk) * 8
			for sb := 0; sb < 8; sb++ {
				scales[idx+sb] = d * sc[sb]
				mins[idx+sb] = dmin * mn[sb]
			}
		}
	}
	return scales, mins
}

func extractQ4KDirect(data []byte, rows, cols int) []int8 {
	bytesPerBlock := 144
	blocksPerRow := cols / 256
	out := make([]int8, rows*cols)
	for row := 0; row < rows; row++ {
		for blk := 0; blk < blocksPerRow; blk++ {
			offset := (row*blocksPerRow + blk) * bytesPerBlock
			qs := data[offset+16 : offset+144]
			base := row*cols + blk*256
			for i := 0; i < 256; i++ {
				if i%2 == 0 {
					out[base+i] = int8(qs[i/2] & 0xf)
				} else {
					out[base+i] = int8(qs[i/2] >> 4)
				}
			}
		}
	}
	return out
}


// avgQ4KScale returns average block scale (uses proper fp16 decode).
func avgQ4KScale(data []byte, rows, cols int) float32 {
	bytesPerBlock := 144
	blocksPerRow := cols / 256
	nBlocks := rows * blocksPerRow
	var sum float32
	for i := 0; i < nBlocks; i++ {
		offset := i * bytesPerBlock
		h := uint16(data[offset]) | uint16(data[offset+1])<<8
		sum += fp16ToFloat(h)
	}
	return sum / float32(nBlocks)
}

func fp16ToFloat(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	if exp == 0 {
		if mant == 0 { return 0 }
		for mant&0x400 == 0 { mant <<= 1; exp-- }
		exp++; mant &= 0x3ff
	} else if exp == 31 {
		return 0 // inf/nan → treat as 0
	}
	exp = exp + (127 - 15)
	bits := (sign << 31) | (exp << 23) | (mant << 13)
	return *(*float32)(unsafe.Pointer(&bits))
}


func main() {
	modelPath := flag.String("model", "", "GGUF model path")
	prompt := flag.String("prompt", "Hello", "prompt")
	nTokens := flag.Int("tokens", 16, "tokens to generate")
	nThreads := flag.Int("threads", 8, "threads")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: ime2run -model <path>")
		os.Exit(1)
	}
	runtime.GOMAXPROCS(*nThreads)

	// Load GGUF
	t0 := time.Now()
	g, err := gguf.Open(*modelPath)
	if err != nil {
		fatal("open: %v", err)
	}

	arch := metaStr(g, "general.architecture", "llama")
	nEmbd := metaInt(g, arch+".embedding_length", 0)
	nHeads := metaInt(g, arch+".attention.head_count", 0)
	nKVHeads := metaInt(g, arch+".attention.head_count_kv", nHeads)
	nLayers := metaInt(g, arch+".block_count", 0)
	nFF := metaInt(g, arch+".feed_forward_length", 0)
	rmsEps := metaF32(g, arch+".attention.layer_norm_rms_epsilon", 1e-5)
	ropeBase := metaF32(g, arch+".rope.freq_base", 10000.0)
	headDim := nEmbd / nHeads
	nQEmbd := nHeads * headDim
	nKVEmbd := nKVHeads * headDim

	// Vocab from token_embd shape
	tokT, _ := g.TensorByName("token_embd.weight")
	nVocab := int(tokT.Shape[1])

	fmt.Fprintf(os.Stderr, "arch=%s embd=%d heads=%d kv=%d layers=%d ff=%d vocab=%d headDim=%d\n",
		arch, nEmbd, nHeads, nKVHeads, nLayers, nFF, nVocab, headDim)

	// Dequant all weights to F32, then quantize to INT8 and pre-pack
	// This is the naive approach (uses 2× memory) but proves the concept
	type layerWeights struct {
		wqPacked, wkPacked, wvPacked, woPacked       []int8
		wqRaw, wkRaw, wvRaw, woRaw                   []int8
		gateRaw, upRaw, downRaw                       []int8
		gatePacked, upPacked, downPacked             []int8
		wqScale, wkScale, wvScale, woScale           float32
		gateScale, upScale, downScale                float32
		// Per-sub-block scales for correct Q4K matmul
		wqScales, wqMins []float32
		wkScales, wkMins []float32
		wvScales, wvMins []float32
		woScales, woMins []float32
		gateScales, gateMins []float32
		upScales, upMins []float32
		downScales, downMins []float32
		attnNorm, ffnNorm                            []float32
		qNorm, kNorm                                 []float32
		wqRows, wkRows, wvRows, woRows              int
		wqCols, wkCols, wvCols, woCols              int
		gateRows, upRows, downRows                   int
		gateCols, upCols, downCols                   int
	}

	layers := make([]layerWeights, nLayers)
	
	// Helper: extract Q4K nibbles directly to INT8 and pre-pack tiles
	packWeight := func(name string, rows, cols int) ([]int8, []int8, float32, []float32, []float32) {
		t, ok := g.TensorByName(name)
		if !ok { fatal("tensor %s not found", name) }
		rowsPad := ((rows + 3) / 4) * 4
		colsPad := ((cols + 7) / 8) * 8
		var i8Pad []int8
		var scale float32
		var scales, mins []float32
		if t.QType == 12 { // Q4_K: direct nibble extraction (correct interleaved order)
			raw, _ := g.Raw(t)
			i8 := extractQ4KDirect(raw, rows, cols)
			i8Pad = make([]int8, rowsPad*colsPad)
			for r := 0; r < rows; r++ { copy(i8Pad[r*colsPad:r*colsPad+cols], i8[r*cols:(r+1)*cols]) }
			scale = avgQ4KScale(raw, rows, cols)
			scales, mins = extractQ4KScales(raw, rows, cols)
		} else {
			scales = nil; mins = nil
			f32, err := g.DequantF32(t)
			if err != nil { fatal("dequant %s: %v", name, err) }
			f32Pad := make([]float32, rowsPad*colsPad)
			for r := 0; r < rows; r++ { copy(f32Pad[r*colsPad:r*colsPad+cols], f32[r*cols:(r+1)*cols]) }
			i8Pad = make([]int8, rowsPad*colsPad)
			scale = inference.QuantizeF32ToINT8(f32Pad, i8Pad)
		}
		packed := ime2.PackTiles(i8Pad, rowsPad, colsPad)
		return packed, i8Pad, scale, scales, mins
	}

	loadNorm := func(name string) []float32 {
		t, ok := g.TensorByName(name)
		if !ok {
			fatal("tensor %s not found", name)
		}
		f32, err := g.DequantF32(t)
		if err != nil {
			fatal("dequant %s: %v", name, err)
		}
		return f32
	}

	tensorName := func(base string, il int) string {
		switch base {
		case "wq":
			return fmt.Sprintf("blk.%d.attn_q.weight", il)
		case "wk":
			return fmt.Sprintf("blk.%d.attn_k.weight", il)
		case "wv":
			return fmt.Sprintf("blk.%d.attn_v.weight", il)
		case "wo":
			return fmt.Sprintf("blk.%d.attn_output.weight", il)
		case "gate":
			return fmt.Sprintf("blk.%d.ffn_gate.weight", il)
		case "up":
			return fmt.Sprintf("blk.%d.ffn_up.weight", il)
		case "down":
			return fmt.Sprintf("blk.%d.ffn_down.weight", il)
		case "attn_norm":
			return fmt.Sprintf("blk.%d.attn_norm.weight", il)
		case "ffn_norm":
			return fmt.Sprintf("blk.%d.ffn_norm.weight", il)
		}
		return ""
	}

	fmt.Fprintf(os.Stderr, "Loading and packing weights...\n")
	for il := 0; il < nLayers; il++ {
		l := &layers[il]
		l.wqRows, l.wqCols = nQEmbd, nEmbd
		l.wkRows, l.wkCols = nKVEmbd, nEmbd
		l.wvRows, l.wvCols = nKVEmbd, nEmbd
		l.woRows, l.woCols = nEmbd, nQEmbd
		l.gateRows, l.gateCols = nFF, nEmbd
		l.upRows, l.upCols = nFF, nEmbd
		l.downRows, l.downCols = nEmbd, nFF

		l.wqPacked, l.wqRaw, l.wqScale, l.wqScales, l.wqMins = packWeight(tensorName("wq", il), l.wqRows, l.wqCols)
		l.wkPacked, l.wkRaw, l.wkScale, l.wkScales, l.wkMins = packWeight(tensorName("wk", il), l.wkRows, l.wkCols)
		l.wvPacked, l.wvRaw, l.wvScale, l.wvScales, l.wvMins = packWeight(tensorName("wv", il), l.wvRows, l.wvCols)
		l.woPacked, l.woRaw, l.woScale, l.woScales, l.woMins = packWeight(tensorName("wo", il), l.woRows, l.woCols)
		l.gatePacked, l.gateRaw, l.gateScale, l.gateScales, l.gateMins = packWeight(tensorName("gate", il), l.gateRows, l.gateCols)
		l.upPacked, l.upRaw, l.upScale, l.upScales, l.upMins = packWeight(tensorName("up", il), l.upRows, l.upCols)
		l.downPacked, l.downRaw, l.downScale, l.downScales, l.downMins = packWeight(tensorName("down", il), l.downRows, l.downCols)
		l.attnNorm = loadNorm(tensorName("attn_norm", il))
		l.ffnNorm = loadNorm(tensorName("ffn_norm", il))
		if qn, ok := g.TensorByName(fmt.Sprintf("blk.%d.attn_q_norm.weight", il)); ok {
			l.qNorm, _ = g.DequantF32(qn)
		}
		if kn, ok := g.TensorByName(fmt.Sprintf("blk.%d.attn_k_norm.weight", il)); ok {
			l.kNorm, _ = g.DequantF32(kn)
		}

		if il%7 == 0 {
			fmt.Fprintf(os.Stderr, "  layer %d/%d loaded\n", il, nLayers)
		}
	}

	// Output norm + token embeddings (for LM head via tied embeddings)
	outputNorm := loadNorm("output_norm.weight")
	tokEmbdF32, _ := g.DequantF32(tokT)


	loadTime := time.Since(t0)
	fmt.Fprintf(os.Stderr, "Loaded in %.1fs\n", loadTime.Seconds())

	// Create persistent worker pool
	pool := ime2.NewWorkerPool(*nThreads)
	defer pool.Close()

	// Tokenize prompt
	tok, _ := gguf.NewTokenizer(g); tok.SetModelPath(*modelPath)
	promptTokens, _ := tok.Encode(*prompt)
	fmt.Fprintf(os.Stderr, "Prompt tokens: %v\n", promptTokens)


	// Pre-allocate reusable buffers to avoid per-step allocation

	// --- Decode loop ---
	// Simplified: no KV cache (recompute attention each time — slow but correct)
	// This proves the pure Go path works end-to-end.

	// Pre-allocate all decode buffers (zero allocation in hot path)
	maxK := pad8(nFF) // 3072 padded = 3072
	maxM := pad4(nFF) // 3072
	_xI8 := make([]int8, maxK)
	_xBroadcast := make([]int8, 4*maxK)
	_actPacked := make([]int8, 4*maxK) // PackTiles output = same size as input for 4 rows
	_resultI32 := make([]int32, maxM*4*2)
	_xn := make([]float32, maxK)
	_xn2 := make([]float32, maxK)
	_qOut := make([]float32, maxM)
	_hidden := make([]float32, maxK)

	// Helper: quantize+pack activation into pre-allocated buffers (zero alloc)
	var _actScale float32
	packAct := func(x []float32, K int) []int8 {
		xI8 := _xI8[:K]
		var maxAbs float32
		for _, v := range x[:K] {
			if v < 0 { v = -v }
			if v > maxAbs { maxAbs = v }
		}
		_actScale = maxAbs / 127.0
		if maxAbs > 0 {
			s := 127.0 / maxAbs
			for i := 0; i < K; i++ {
				v := x[i] * s
				if v > 127 { v = 127 } else if v < -128 { v = -128 }
				xI8[i] = int8(v)
			}
		}
		// Broadcast 4×
		bc := _xBroadcast[:4*K]
		copy(bc[0:K], xI8)
		copy(bc[K:2*K], xI8)
		copy(bc[2*K:3*K], xI8)
		copy(bc[3*K:4*K], xI8)
		// Pack tiles in-place
		packed := _actPacked[:4*K]
		for rg := 0; rg < 4; rg += 4 {
			for ki := 0; ki < K; ki += 8 {
				tileIdx := ki / 8
				tileBase := tileIdx * 32
				for r := 0; r < 4; r++ {
					copy(packed[tileBase+r*8:tileBase+r*8+8], bc[r*K+ki:r*K+ki+8])
				}
			}
		}
		return packed
	}
	_ = packAct; _ = _resultI32; _ = _xn; _ = _xn2; _ = _qOut; _ = _hidden


	// KV cache: [nLayers][nKVHeads * headDim * nCtx] for both K and V
	nCtx := 512
	kvSize := nKVHeads * headDim * nCtx
	kCache := make([][]float32, nLayers)
	vCache := make([][]float32, nLayers)
	for il := range kCache {
		kCache[il] = make([]float32, kvSize)
		vCache[il] = make([]float32, kvSize)
	}
	nPast := 0
	allTokens := promptTokens
	t1 := time.Now()

	for step := 0; step < len(promptTokens)+*nTokens-1; step++ {
		tokID := allTokens[step]

		// Embedding lookup
		x := make([]float32, nEmbd)
		copy(x, tokEmbdF32[tokID*nEmbd:(tokID+1)*nEmbd])

		// Layer loop (simplified: no attention, just FFN for speed test)
		for il := 0; il < nLayers; il++ {
			l := &layers[il]

		// === Attention with KV cache ===
		xn := _xn[:nEmbd]
		inference.RMSNorm(x, l.attnNorm, xn, rmsEps)
		K1 := pad8(l.wqCols)
		for i := nEmbd; i < K1; i++ { _xn[i] = 0 }
		actP := packAct(_xn[:K1], K1)

		// Q, K, V projections (per-sub-block scale correction for Q4K)
		qF := make([]float32, l.wqRows)
		if l.wqScales != nil {
			matVecQ4KCorrect(l.wqRows, l.wqCols, l.wqRaw, l.wqScales, l.wqMins, xn[:l.wqCols], qF)
		} else {
			qRes := _resultI32[:pad4(l.wqRows)*4]
			ime2.GemmINT8Packed(pad4(l.wqRows), 4, K1, l.wqPacked, actP, qRes)
			for i := range qF { qF[i] = float32(qRes[i*4]) * l.wqScale * _actScale }
		}
		kF := make([]float32, l.wkRows)
		if l.wkScales != nil {
			matVecQ4KCorrect(l.wkRows, l.wkCols, l.wkRaw, l.wkScales, l.wkMins, xn[:l.wkCols], kF)
		} else {
			kRes := _resultI32[:pad4(l.wkRows)*4]
			ime2.GemmINT8Packed(pad4(l.wkRows), 4, K1, l.wkPacked, actP, kRes)
			for i := range kF { kF[i] = float32(kRes[i*4]) * l.wkScale * _actScale }
		}
		vF := make([]float32, l.wvRows)
		if l.wvScales != nil {
			matVecQ4KCorrect(l.wvRows, l.wvCols, l.wvRaw, l.wvScales, l.wvMins, xn[:l.wvCols], vF)
		} else {
			vRes := _resultI32[:pad4(l.wvRows)*4]
			ime2.GemmINT8Packed(pad4(l.wvRows), 4, K1, l.wvPacked, actP, vRes)
			for i := range vF { vF[i] = float32(vRes[i*4]) * l.wvScale * _actScale }
		}

		// Apply K norm, then RoPE, then store in cache
		pos := nPast
		nKVD := nKVHeads * headDim
		if l.kNorm != nil {
			for kh := 0; kh < nKVHeads; kh++ {
				var ss float32
				for d := 0; d < headDim; d++ { ss += kF[kh*headDim+d] * kF[kh*headDim+d] }
				ss = float32(1.0 / math.Sqrt(float64(ss/float32(headDim)+rmsEps)))
				for d := 0; d < headDim; d++ { kF[kh*headDim+d] = kF[kh*headDim+d] * ss * l.kNorm[d] }
			}
		}
		copy(kCache[il][pos*nKVD:pos*nKVD+nKVD], kF)
		copy(vCache[il][pos*nKVD:pos*nKVD+nKVD], vF)
		// Apply RoPE to K at current position
		for kh := 0; kh < nKVHeads; kh++ {
			applyRoPE(kCache[il][pos*nKVD+kh*headDim:pos*nKVD+(kh+1)*headDim], headDim, pos, ropeBase)
		}

		// Attention: Q heads attend over cached K/V
		repFactor := nHeads / nKVHeads
		attnOut := _qOut[:nHeads*headDim]
		invSqrtD := float32(1.0 / math.Sqrt(float64(headDim)))
		for h := 0; h < nHeads; h++ {
			kvH := h / repFactor
			// Q head: QK norm + RoPE
			qHead := make([]float32, headDim)
			copy(qHead, qF[h*headDim:(h+1)*headDim])
			if l.qNorm != nil {
				// RMS norm on head dimension
				var ss float32
				for d := 0; d < headDim; d++ { ss += qHead[d] * qHead[d] }
				ss = float32(1.0 / math.Sqrt(float64(ss/float32(headDim)+rmsEps)))
				for d := 0; d < headDim; d++ { qHead[d] = qHead[d] * ss * l.qNorm[d] }
			}
			applyRoPE(qHead, headDim, pos, ropeBase)
			// Compute scores over [0..pos]
			maxScore := float32(-1e30)
			scores := make([]float32, pos+1)
			for t := 0; t <= pos; t++ {
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += qHead[d] * kCache[il][t*nKVD+kvH*headDim+d]
				}
				scores[t] = dot * invSqrtD
				if scores[t] > maxScore { maxScore = scores[t] }
			}
			// Softmax
			var sumExp float32
			for i := range scores { scores[i] = float32(math.Exp(float64(scores[i] - maxScore))); sumExp += scores[i] }
			for i := range scores { scores[i] /= sumExp }
			// Weighted V sum
			for d := 0; d < headDim; d++ {
				var sum float32
				for t := 0; t <= pos; t++ {
					sum += scores[t] * vCache[il][t*nKVD+kvH*headDim+d]
				}
				attnOut[h*headDim+d] = sum
			}
		}

		// Output projection
		K2 := pad8(l.woCols)
		for i := nHeads*headDim; i < K2; i++ { _qOut[i] = 0 }
		actP2 := packAct(_qOut[:K2], K2)
		woRes := _resultI32[:pad4(l.woRows)*4]
		ime2.GemmINT8Packed(pad4(l.woRows), 4, K2, l.woPacked, actP2, woRes)
		for i := 0; i < nEmbd; i++ { x[i] += float32(woRes[i*4]) * l.woScale * _actScale }

		xn2 := _xn2[:nEmbd]
		inference.RMSNorm(x, l.ffnNorm, xn2, rmsEps)
		K3 := pad8(l.gateCols)
		for i := nEmbd; i < K3; i++ { _xn2[i] = 0 }
		actPF := packAct(_xn2[:K3], K3)
		gateF := make([]float32, l.gateRows)
		if l.gateScales != nil {
			matVecQ4KCorrect(l.gateRows, l.gateCols, l.gateRaw, l.gateScales, l.gateMins, xn2[:l.gateCols], gateF)
		} else {
			gRes := _resultI32[:pad4(l.gateRows)*4]
			ime2.GemmINT8Packed(pad4(l.gateRows), 4, K3, l.gatePacked, actPF, gRes)
			for i := range gateF { gateF[i] = float32(gRes[i*4]) * l.gateScale * _actScale }
		}
		upF := make([]float32, l.upRows)
		if l.upScales != nil {
			matVecQ4KCorrect(l.upRows, l.upCols, l.upRaw, l.upScales, l.upMins, xn2[:l.upCols], upF)
		} else {
			uRes := _resultI32[:pad4(l.upRows)*4]
			ime2.GemmINT8Packed(pad4(l.upRows), 4, K3, l.upPacked, actPF, uRes)
			for i := range upF { upF[i] = float32(uRes[i*4]) * l.upScale * _actScale }
		}
		hidden := _hidden[:nFF]
		for i := 0; i < nFF; i++ {
			hidden[i] = silu(gateF[i]) * upF[i]
		}
		K4 := pad8(l.downCols)
		for i := nFF; i < K4; i++ { _hidden[i] = 0 }
		actPD := packAct(_hidden[:K4], K4)
		downF := make([]float32, l.downRows)
		if l.downScales != nil {
			matVecQ4KCorrect(l.downRows, l.downCols, l.downRaw, l.downScales, l.downMins, hidden[:l.downCols], downF)
		} else {
			dRes := _resultI32[:pad4(l.downRows)*4]
			ime2.GemmINT8Packed(pad4(l.downRows), 4, K4, l.downPacked, actPD, dRes)
			for i := range downF { downF[i] = float32(dRes[i*4]) * l.downScale * _actScale }
		}
		for i := 0; i < nEmbd; i++ { x[i] += downF[i] }

		}

		// Output norm + LM head
		xn := make([]float32, nEmbd)
		inference.RMSNorm(x, outputNorm, xn, rmsEps)

		// LM head via vmadot (pre-packed tok_embd)
		logits := make([]float32, nVocab)
		for v := 0; v < nVocab; v++ {
			var sum float32
			for k := 0; k < nEmbd; k++ { sum += tokEmbdF32[v*nEmbd+k] * xn[k] }
			logits[v] = sum
		}

		// Argmax
		nextTok := 0
		maxVal := logits[0]
		for i := 1; i < nVocab; i++ {
			if logits[i] > maxVal {
				maxVal = logits[i]
				nextTok = i
			}
		}

		// If in prefill phase, just continue
		if step < len(promptTokens)-1 {
			nPast++
			continue
		}

		// First generated token or subsequent
		if step == len(promptTokens)-1 {
			prefillTime := time.Since(t1)
			fmt.Fprintf(os.Stderr, "Prefill: %.3fs (%.1f tok/s)\n",
				prefillTime.Seconds(), float64(len(promptTokens))/prefillTime.Seconds())
			fmt.Fprintf(os.Stderr, "logits[0]=%.4f max_idx=%d val=%.4f\n", logits[0], nextTok, maxVal)
			t1 = time.Now()
		}

		nPast++
		allTokens = append(allTokens, nextTok)
	}

	decodeTime := time.Since(t1)
	genCount := len(allTokens) - len(promptTokens)
	fmt.Fprintf(os.Stderr, "Decode: %.3fs (%.2f tok/s, %d tokens)\n",
		decodeTime.Seconds(), float64(genCount)/decodeTime.Seconds(), genCount)

	// Decode output
	output := tok.Decode(allTokens)
	fmt.Printf("Output: %s\n", output)
}

func pad4(n int) int  { return ((n + 3) / 4) * 4 }
func pad8(n int) int  { return ((n + 7) / 8) * 8 }
func padF32(src []float32, n int) []float32 {
	if len(src) >= n {
		return src[:n]
	}
	dst := make([]float32, n)
	copy(dst, src)
	return dst
}


// applyRoPE applies Rotary Position Embedding to a vector of headDim floats.

// matVecQ4KCorrect performs out[M] = W_q4k[M×K] · act[K] with per-sub-block scale correction.
// wData: raw Q4_K bytes, wPacked: pre-extracted INT8 nibbles (pre-packed in tiles),
// wScales/wMins: per sub-block scales [M × blocksPerRow × 8]
// act: F32 activation, out: F32 output
func matVecQ4KCorrect(M, K int, wPacked []int8, wScales, wMins []float32, act []float32, out []float32) {
	// Quantize activation to INT8
	actI8 := make([]int8, K)
	var maxAbs float32
	for _, v := range act[:K] {
		a := v; if a < 0 { a = -a }
		if a > maxAbs { maxAbs = a }
	}
	actScale := float32(0)
	if maxAbs > 0 {
		actScale = 127.0 / maxAbs
		for i := 0; i < K; i++ {
			v := act[i] * actScale
			if v > 127 { v = 127 } else if v < -128 { v = -128 }
			actI8[i] = int8(v)
		}
	}
	actDeScale := float32(0)
	if actScale > 0 { actDeScale = 1.0 / actScale }

	blocksPerRow := K / 256
	subsPerRow := blocksPerRow * 8 // 8 sub-blocks per Q4_K block

	// For each output row: accumulate dot products per sub-block with individual scales
	for row := 0; row < M; row++ {
		var sum float32
		for sb := 0; sb < subsPerRow; sb++ {
			// Sub-block: 32 elements starting at row*K + sb*32
			elemOff := sb * 32
			// Dot product (scalar, can optimize later with vmadot)
			var dot int32
			for i := 0; i < 32; i++ {
				dot += int32(wPacked[row*K+elemOff+i]) * int32(actI8[elemOff+i])
			}
			// Apply per-sub-block scale
			wScale := wScales[row*subsPerRow+sb]
			wMin := wMins[row*subsPerRow+sb]
			sum += float32(dot) * wScale * actDeScale
			// Min correction: each nibble value has dmin*min subtracted
			// The dot product of act with the constant -min offset:
			var actSum int32
			for i := 0; i < 32; i++ {
				actSum += int32(actI8[elemOff+i])
			}
			sum -= float32(actSum) * wMin * actDeScale
		}
		out[row] = sum
	}
}

func applyRoPE(x []float32, headDim int, pos int, ropeBase float32) {
	for i := 0; i < headDim/2; i++ {
		freq := 1.0 / math.Pow(float64(ropeBase), float64(2*i)/float64(headDim))
		theta := float64(pos) * freq
		cos_t := float32(math.Cos(theta))
		sin_t := float32(math.Sin(theta))
		x0 := x[2*i]
		x1 := x[2*i+1]
		x[2*i] = x0*cos_t - x1*sin_t
		x[2*i+1] = x0*sin_t + x1*cos_t
	}
}

func silu(x float32) float32 {
	return x / (1.0 + float32(math.Exp(-float64(x))))
}

func metaStr(g *gguf.GGUF, key, def string) string {
	if v, ok := g.MetaString(key); ok {
		return v
	}
	return def
}
func metaInt(g *gguf.GGUF, key string, def int) int {
	if v, ok := g.MetaUint32(key); ok {
		return int(v)
	}
	return def
}
func metaF32(g *gguf.GGUF, key string, def float32) float32 {
	if v, ok := g.MetaFloat32(key); ok {
		return v
	}
	return def
}
func fatal(f string, a ...any) {
	fmt.Fprintf(os.Stderr, f+"\n", a...)
	os.Exit(1)
}

var _ = unsafe.Pointer(nil)
