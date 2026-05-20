package lmhead

// Chunked LM head: processes vocab projection in GPU-sized chunks.
//
// When full LM head weights do not fit in VRAM, split into chunks:
//  1. Allocate a GPU buffer for chunkRows rows
//  2. Upload chunk of LM head weights
//  3. GPU GEMV for chunkRows rows
//  4. Download logits for those rows
//  5. Repeat for all chunks
//
// This trades upload bandwidth for GPU compute speed.
func ChunkedLMHead(logits, hidden, lmHead []float32, vocabSize, h int) bool {
	if vocabSize <= 0 || h <= 0 || len(logits) < vocabSize || len(hidden) < h {
		return false
	}
	maxInt := int(^uint(0) >> 1)
	if vocabSize > maxInt/h || len(lmHead) < vocabSize*h {
		return false
	}
	free, _ := MemInfo()
	if free < 64*1024*1024 { // need at least 64MB free
		return false
	}

	if free > uint64(maxInt) {
		free = uint64(maxInt)
	}
	usable := int(free) - 32*1024*1024
	if h > maxInt/4 || usable <= 0 {
		return false
	}
	chunkRows := usable / (h * 4)
	if chunkRows < 1024 {
		return false
	}
	if chunkRows > vocabSize {
		chunkRows = vocabSize
	}

	chunkElems, ok := checkedMulInt(chunkRows, h)
	if !ok {
		return false
	}

	wBuf := NewDevBuf(chunkElems)
	defer wBuf.Free()
	if err := wBuf.ToGPU(); err != nil {
		return false
	}
	outBuf := NewDevBuf(chunkRows)
	defer outBuf.Free()
	if err := outBuf.ToGPU(); err != nil {
		return false
	}
	inBuf := NewDevBuf(h)
	defer inBuf.Free()
	copy(inBuf.Data(), hidden[:h])
	inBuf.MarkDirty()
	if err := inBuf.ToGPU(); err != nil {
		return false
	}

	for start := 0; start < vocabSize; start += chunkRows {
		end := start + chunkRows
		if end > vocabSize {
			end = vocabSize
		}
		rows := end - start

		wData := wBuf.Data()
		copy(wData[:rows*h], lmHead[start*h:end*h])
		wBuf.MarkDirty()
		if err := wBuf.ToGPU(); err != nil {
			return false
		}

		if rows == chunkRows {
			DevLMHead(outBuf, inBuf, wBuf, rows, h)
		} else {
			outSlice := outBuf.Slice(0, rows)
			wSlice := wBuf.Slice(0, rows*h)
			DevLMHead(outSlice, inBuf, wSlice, rows, h)
		}
		Sync()

		outData := outBuf.Data()
		copy(logits[start:end], outData[:rows])
	}

	return true
}
