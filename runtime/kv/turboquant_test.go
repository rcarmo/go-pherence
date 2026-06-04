package kv

import (
	"math"
	"testing"
)

func TestTurboQuantKeyValueMethodsMatchGenericVectorAPI(t *testing.T) {
	tq := NewTurboQuantState(16, 4, TurboQuantConfig{KeyBits: 4, ValueBits: 2, ResidualWindow: 0})
	vec := make([]float32, tq.HeadDim)
	for i := range vec {
		vec[i] = float32(math.Sin(float64(i)*0.23) + math.Cos(float64(i+2)*0.17))
	}

	pk, minK, scaleK := tq.QuantizeKey(vec)
	wantPK, wantMinK, wantScaleK := tq.QuantizeVector(vec, tq.RotationK, tq.CodebookK, tq.Config.KeyBits)
	if string(pk) != string(wantPK) || minK != wantMinK || scaleK != wantScaleK {
		t.Fatalf("QuantizeKey mismatch got=(%v,%v,%v) want=(%v,%v,%v)", pk, minK, scaleK, wantPK, wantMinK, wantScaleK)
	}
	gotK := tq.DequantizeKey(pk, minK, scaleK, tq.HeadDim)
	wantK := tq.DequantizeVector(wantPK, wantMinK, wantScaleK, tq.RotationK, tq.Config.KeyBits, tq.HeadDim)
	for i := range gotK {
		if gotK[i] != wantK[i] {
			t.Fatalf("DequantizeKey[%d]=%v want %v", i, gotK[i], wantK[i])
		}
	}

	pv, minV, scaleV := tq.QuantizeValue(vec)
	wantPV, wantMinV, wantScaleV := tq.QuantizeVector(vec, tq.RotationV, tq.CodebookV, tq.Config.ValueBits)
	if string(pv) != string(wantPV) || minV != wantMinV || scaleV != wantScaleV {
		t.Fatalf("QuantizeValue mismatch got=(%v,%v,%v) want=(%v,%v,%v)", pv, minV, scaleV, wantPV, wantMinV, wantScaleV)
	}
	gotV := tq.DequantizeValue(pv, minV, scaleV, tq.HeadDim)
	wantV := tq.DequantizeVector(wantPV, wantMinV, wantScaleV, tq.RotationV, tq.Config.ValueBits, tq.HeadDim)
	for i := range gotV {
		if gotV[i] != wantV[i] {
			t.Fatalf("DequantizeValue[%d]=%v want %v", i, gotV[i], wantV[i])
		}
	}
}

func TestPackIndicesTo(t *testing.T) {
	indices := []byte{1, 2, 3, 4, 5, 6, 7}
	want := packIndices(indices, 4)
	dst := bytesWithSentinel(len(want), 0xAA)
	if !packIndicesTo(dst, indices, 4) {
		t.Fatal("packIndicesTo rejected valid input")
	}
	if string(dst) != string(want) {
		t.Fatalf("packed mismatch got=%v want=%v", dst, want)
	}
	if packIndicesTo(dst[:len(dst)-1], indices, 4) {
		t.Fatal("packIndicesTo accepted short dst")
	}
}

func bytesWithSentinel(n int, v byte) []byte {
	out := make([]byte, n)
	for i := range out {
		out[i] = v
	}
	return out
}

func TestTurboQuantKeyValueQuantizeWithScratch(t *testing.T) {
	tq := NewTurboQuantState(8, 2, TurboQuantConfig{KeyBits: 4, ValueBits: 2, ResidualWindow: 0})
	vec := []float32{0.5, -1, 1.5, 2, -0.25, 0.75, -1.25, 0.125}
	rotated := make([]float32, tq.HeadDim)
	indices := make([]byte, tq.HeadDim)
	pk, minK, scaleK, ok := tq.QuantizeKeyWithScratch(vec, rotated, indices)
	if !ok {
		t.Fatal("QuantizeKeyWithScratch rejected valid input")
	}
	wantPK, wantMinK, wantScaleK := tq.QuantizeKey(vec)
	if string(pk) != string(wantPK) || minK != wantMinK || scaleK != wantScaleK {
		t.Fatalf("QuantizeKeyWithScratch mismatch got=(%v,%v,%v) want=(%v,%v,%v)", pk, minK, scaleK, wantPK, wantMinK, wantScaleK)
	}
	pv, minV, scaleV, ok := tq.QuantizeValueWithScratch(vec, rotated, indices)
	if !ok {
		t.Fatal("QuantizeValueWithScratch rejected valid input")
	}
	wantPV, wantMinV, wantScaleV := tq.QuantizeValue(vec)
	if string(pv) != string(wantPV) || minV != wantMinV || scaleV != wantScaleV {
		t.Fatalf("QuantizeValueWithScratch mismatch got=(%v,%v,%v) want=(%v,%v,%v)", pv, minV, scaleV, wantPV, wantMinV, wantScaleV)
	}
	if _, _, _, ok := tq.QuantizeKeyWithScratch(vec, rotated[:tq.HeadDim-1], indices); ok {
		t.Fatal("QuantizeKeyWithScratch accepted short rotated scratch")
	}
	if _, _, _, ok := tq.QuantizeValueWithScratch(vec, rotated, indices[:tq.HeadDim-1]); ok {
		t.Fatal("QuantizeValueWithScratch accepted short index scratch")
	}
	keyDst := bytesWithSentinel(len(wantPK), 0xAA)
	minKTo, scaleKTo, ok := tq.QuantizeKeyTo(keyDst, vec, rotated, indices)
	if !ok || string(keyDst) != string(wantPK) || minKTo != wantMinK || scaleKTo != wantScaleK {
		t.Fatalf("QuantizeKeyTo mismatch ok=%v got=(%v,%v,%v) want=(%v,%v,%v)", ok, keyDst, minKTo, scaleKTo, wantPK, wantMinK, wantScaleK)
	}
	valueDst := bytesWithSentinel(len(wantPV), 0xAA)
	minVTo, scaleVTo, ok := tq.QuantizeValueTo(valueDst, vec, rotated, indices)
	if !ok || string(valueDst) != string(wantPV) || minVTo != wantMinV || scaleVTo != wantScaleV {
		t.Fatalf("QuantizeValueTo mismatch ok=%v got=(%v,%v,%v) want=(%v,%v,%v)", ok, valueDst, minVTo, scaleVTo, wantPV, wantMinV, wantScaleV)
	}
	if _, _, ok := tq.QuantizeKeyTo(keyDst[:len(keyDst)-1], vec, rotated, indices); ok {
		t.Fatal("QuantizeKeyTo accepted short packed dst")
	}
}

func TestTurboQuantKeyValueDequantizeTo(t *testing.T) {
	tq := NewTurboQuantState(8, 2, TurboQuantConfig{KeyBits: 4, ValueBits: 2, ResidualWindow: 0})
	vec := []float32{0.5, -1, 1.5, 2, -0.25, 0.75, -1.25, 0.125}
	pk, minK, scaleK := tq.QuantizeKey(vec)
	wantK := tq.DequantizeKey(pk, minK, scaleK, tq.HeadDim)
	dstK := make([]float32, tq.HeadDim)
	if !tq.DequantizeKeyTo(dstK, pk, minK, scaleK, tq.HeadDim) {
		t.Fatal("DequantizeKeyTo rejected valid input")
	}
	for i := range dstK {
		if dstK[i] != wantK[i] {
			t.Fatalf("DequantizeKeyTo[%d]=%v want %v", i, dstK[i], wantK[i])
		}
	}
	pv, minV, scaleV := tq.QuantizeValue(vec)
	wantV := tq.DequantizeValue(pv, minV, scaleV, tq.HeadDim)
	dstV := make([]float32, tq.HeadDim)
	if !tq.DequantizeValueTo(dstV, pv, minV, scaleV, tq.HeadDim) {
		t.Fatal("DequantizeValueTo rejected valid input")
	}
	for i := range dstV {
		if dstV[i] != wantV[i] {
			t.Fatalf("DequantizeValueTo[%d]=%v want %v", i, dstV[i], wantV[i])
		}
	}
	if tq.DequantizeKeyTo(dstK[:tq.HeadDim-1], pk, minK, scaleK, tq.HeadDim) {
		t.Fatal("DequantizeKeyTo accepted short dst")
	}
	if tq.DequantizeValueTo(dstV[:tq.HeadDim-1], pv, minV, scaleV, tq.HeadDim) {
		t.Fatal("DequantizeValueTo accepted short dst")
	}
}

func TestTurboQuantKeyValueMethodsHandleNilState(t *testing.T) {
	var tq *TurboQuantState
	if p, vMin, scale := tq.QuantizeKey([]float32{1, 2}); p != nil || vMin != 0 || scale != 0 {
		t.Fatalf("nil QuantizeKey=(%v,%v,%v)", p, vMin, scale)
	}
	if p, vMin, scale := tq.QuantizeValue([]float32{1, 2}); p != nil || vMin != 0 || scale != 0 {
		t.Fatalf("nil QuantizeValue=(%v,%v,%v)", p, vMin, scale)
	}
	if got := tq.DequantizeKey(nil, 0, 0, 3); len(got) != 3 {
		t.Fatalf("nil DequantizeKey len=%d want 3", len(got))
	}
	if got := tq.DequantizeValue(nil, 0, 0, 3); len(got) != 3 {
		t.Fatalf("nil DequantizeValue len=%d want 3", len(got))
	}
}

func TestTurboQuantSIMDRotationParity(t *testing.T) {
	dim := 17
	vec := make([]float32, dim)
	rot := make([]float32, dim*dim)
	for i := range vec {
		vec[i] = float32(math.Sin(float64(i+1)*0.37) + math.Cos(float64(i)*0.11))
	}
	for i := range rot {
		rot[i] = float32(math.Sin(float64(i+3)*0.13) * 0.25)
	}

	gotRows := make([]float32, dim)
	if !rotateRows(gotRows, vec, rot, dim) {
		t.Fatal("rotateRows rejected valid input")
	}
	gotCols := make([]float32, dim)
	if !rotateCols(gotCols, vec, rot, dim) {
		t.Fatal("rotateCols rejected valid input")
	}

	for i := 0; i < dim; i++ {
		var wantRow, wantCol float32
		for j := 0; j < dim; j++ {
			wantRow += rot[i*dim+j] * vec[j]
			wantCol += rot[j*dim+i] * vec[j]
		}
		if d := math.Abs(float64(gotRows[i] - wantRow)); d > 1e-4 {
			t.Fatalf("rotateRows[%d]=%g want %g diff=%g", i, gotRows[i], wantRow, d)
		}
		if d := math.Abs(float64(gotCols[i] - wantCol)); d > 1e-4 {
			t.Fatalf("rotateCols[%d]=%g want %g diff=%g", i, gotCols[i], wantCol, d)
		}
	}
}

func TestTurboQuantInverseUsesStoredTranspose(t *testing.T) {
	tq := NewTurboQuantState(8, 2, DefaultTurboQuantConfig())
	if len(tq.RotationKT) != len(tq.RotationK) || len(tq.RotationVT) != len(tq.RotationV) {
		t.Fatalf("missing transposed rotations: k=%d kt=%d v=%d vt=%d", len(tq.RotationK), len(tq.RotationKT), len(tq.RotationV), len(tq.RotationVT))
	}
	if got := tq.transposeForRotation(tq.RotationK); len(got) != len(tq.RotationKT) || &got[0] != &tq.RotationKT[0] {
		t.Fatal("key rotation did not select stored transpose")
	}
	if got := tq.transposeForRotation(tq.RotationV); len(got) != len(tq.RotationVT) || &got[0] != &tq.RotationVT[0] {
		t.Fatal("value rotation did not select stored transpose")
	}
	copyRot := append([]float32(nil), tq.RotationK...)
	if got := tq.transposeForRotation(copyRot); got != nil {
		t.Fatalf("external rotation copy selected internal transpose len=%d", len(got))
	}
	for r := 0; r < tq.HeadDim; r++ {
		for c := 0; c < tq.HeadDim; c++ {
			if tq.RotationKT[c*tq.HeadDim+r] != tq.RotationK[r*tq.HeadDim+c] {
				t.Fatalf("bad K transpose at r=%d c=%d", r, c)
			}
			if tq.RotationVT[c*tq.HeadDim+r] != tq.RotationV[r*tq.HeadDim+c] {
				t.Fatalf("bad V transpose at r=%d c=%d", r, c)
			}
		}
	}
}

func TestCompressedKVCacheCompressionUsesScratch(t *testing.T) {
	tq := NewTurboQuantState(4, 1, TurboQuantConfig{KeyBits: 4, ValueBits: 2, ResidualWindow: 0})
	c := NewCompressedKVCache(4, 1, 4, tq, false)
	c.Append([]float32{1, 2, 3, 4}, []float32{4, 3, 2, 1})
	if c.CompressedCount() != 1 || len(c.quantRotated) != 4 || len(c.quantIndices) != 4 {
		t.Fatalf("compression scratch not initialized compressed=%d rotated=%d indices=%d", c.CompressedCount(), len(c.quantRotated), len(c.quantIndices))
	}
	if got, want := len(c.CompressedK[0].Packed), 2; got != want {
		t.Fatalf("key packed len=%d want %d", got, want)
	}
	if got, want := len(c.CompressedV[0].Packed), 1; got != want {
		t.Fatalf("value packed len=%d want %d", got, want)
	}
	rotPtr := &c.quantRotated[0]
	idxPtr := &c.quantIndices[0]
	c.Append([]float32{2, 3, 4, 5}, []float32{5, 4, 3, 2})
	if c.CompressedCount() != 2 || &c.quantRotated[0] != rotPtr || &c.quantIndices[0] != idxPtr {
		t.Fatalf("compression scratch not reused compressed=%d", c.CompressedCount())
	}
}

func TestCompressedKVCacheDirectScratchDecompression(t *testing.T) {
	tq := NewTurboQuantState(4, 1, TurboQuantConfig{KeyBits: 4, ValueBits: 2, ResidualWindow: 1})
	c := NewCompressedKVCache(4, 1, 4, tq, false)
	for i := 0; i < 3; i++ {
		c.Append([]float32{float32(i), float32(i + 1), float32(i + 2), float32(i + 3)}, []float32{float32(-i), float32(i) + 0.5, float32(i) + 1.5, float32(i) + 2.5})
	}
	if c.CompressedCount() != 2 || c.FullCount() != 1 || c.SeqLen() != 3 {
		t.Fatalf("unexpected counts compressed=%d full=%d seq=%d", c.CompressedCount(), c.FullCount(), c.SeqLen())
	}
	gotK := c.GetK()
	gotV := c.GetV()
	if len(gotK) != 12 || len(gotV) != 12 {
		t.Fatalf("unexpected decompressed lengths k=%d v=%d", len(gotK), len(gotV))
	}
	if len(c.scratchK) != 12 || len(c.scratchV) != 12 {
		t.Fatalf("scratch not sized to sequence: k=%d v=%d", len(c.scratchK), len(c.scratchV))
	}
	for i := range gotK {
		if math.IsNaN(float64(gotK[i])) || math.IsInf(float64(gotK[i]), 0) || math.IsNaN(float64(gotV[i])) || math.IsInf(float64(gotV[i]), 0) {
			t.Fatalf("bad decompressed value at %d k=%v v=%v", i, gotK[i], gotV[i])
		}
	}
	gotK2 := c.GetK()
	if len(gotK2) != 12 || &gotK2[0] != &c.scratchK[0] {
		t.Fatalf("GetK did not reuse scratch")
	}
}

func TestCompressedKVCacheUsesSIMDTransposeDecompression(t *testing.T) {
	tq := NewTurboQuantState(4, 1, TurboQuantConfig{KeyBits: 4, ValueBits: 2, ResidualWindow: 0})
	c := NewCompressedKVCache(4, 1, 4, tq, false)
	k := []float32{0.25, -0.5, 1.25, 2.0}
	v := []float32{-1.0, 0.75, 1.5, -0.25}
	c.Append(k, v)
	if c.CompressedCount() != 1 || c.FullCount() != 0 || c.SeqLen() != 1 {
		t.Fatalf("unexpected cache counts: compressed=%d full=%d seq=%d", c.CompressedCount(), c.FullCount(), c.SeqLen())
	}
	gotK := c.GetK()
	gotV := c.GetV()
	if len(gotK) != len(k) || len(gotV) != len(v) {
		t.Fatalf("unexpected decompressed lengths: k=%d v=%d", len(gotK), len(gotV))
	}
	for i := range gotK {
		if math.IsNaN(float64(gotK[i])) || math.IsInf(float64(gotK[i]), 0) {
			t.Fatalf("bad K[%d]=%v", i, gotK[i])
		}
		if math.IsNaN(float64(gotV[i])) || math.IsInf(float64(gotV[i]), 0) {
			t.Fatalf("bad V[%d]=%v", i, gotV[i])
		}
	}
}

func TestTurboQuantRotationRejectsMalformedInputs(t *testing.T) {
	out := make([]float32, 4)
	vec := make([]float32, 4)
	rot := make([]float32, 15)
	if rotateRows(out, vec, rot, 4) {
		t.Fatal("rotateRows accepted short rotation")
	}
	if rotateCols(out, vec, rot, 4) {
		t.Fatal("rotateCols accepted short rotation")
	}
	if rotateRows(out[:3], vec, make([]float32, 16), 4) {
		t.Fatal("rotateRows accepted short output")
	}
	if rotateCols(out, vec[:3], make([]float32, 16), 4) {
		t.Fatal("rotateCols accepted short vector")
	}
}

func TestTurboQuantRoundtrip(t *testing.T) {
	dim := 128
	tq := NewTurboQuantState(dim, 28, DefaultTurboQuantConfig())

	// Create a test vector
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = float32(math.Sin(float64(i)*0.1)) * 2.0
	}

	// Quantize keys (4-bit)
	packed, vMin, scale := tq.QuantizeVector(vec, tq.RotationK, tq.CodebookK, tq.Config.KeyBits)
	restored := tq.DequantizeVector(packed, vMin, scale, tq.RotationK, tq.Config.KeyBits, dim)

	// Measure reconstruction error
	var maxErr, sumErr float64
	for i := range vec {
		d := math.Abs(float64(vec[i] - restored[i]))
		sumErr += d
		if d > maxErr {
			maxErr = d
		}
	}
	meanErr := sumErr / float64(dim)
	t.Logf("4-bit key roundtrip: maxErr=%.6f meanErr=%.6f", maxErr, meanErr)
	t.Logf("  original[0:5]: %v", vec[:5])
	t.Logf("  restored[0:5]: %v", restored[:5])

	// 4-bit should have reasonable error (< 0.5 for values in [-2, 2])
	if maxErr > 1.0 {
		t.Errorf("4-bit key error too large: maxErr=%.6f", maxErr)
	}

	// Quantize values (2-bit)
	packed2, vMin2, scale2 := tq.QuantizeVector(vec, tq.RotationV, tq.CodebookV, tq.Config.ValueBits)
	restored2 := tq.DequantizeVector(packed2, vMin2, scale2, tq.RotationV, tq.Config.ValueBits, dim)

	var maxErr2, sumErr2 float64
	for i := range vec {
		d := math.Abs(float64(vec[i] - restored2[i]))
		sumErr2 += d
		if d > maxErr2 {
			maxErr2 = d
		}
	}
	meanErr2 := sumErr2 / float64(dim)
	t.Logf("2-bit value roundtrip: maxErr=%.6f meanErr=%.6f", maxErr2, meanErr2)

	// 2-bit should have larger but bounded error
	if maxErr2 > 3.0 {
		t.Errorf("2-bit value error too large: maxErr=%.6f", maxErr2)
	}
}

func TestTurboQuantCompressionRatio(t *testing.T) {
	dim := 128

	// Original: 128 × 4 bytes = 512 bytes per vector
	origBytes := dim * 4

	// 4-bit: 128 × 4 bits / 8 = 64 bytes + 4 bytes norm = 68 bytes
	bits4Bytes := (dim*4+7)/8 + 4
	// 2-bit: 128 × 2 bits / 8 = 32 bytes + 4 bytes norm = 36 bytes
	bits2Bytes := (dim*2+7)/8 + 4

	t.Logf("Compression ratios for dim=%d:", dim)
	t.Logf("  Original: %d bytes", origBytes)
	t.Logf("  4-bit key: %d bytes (%.1f×)", bits4Bytes, float64(origBytes)/float64(bits4Bytes))
	t.Logf("  2-bit val: %d bytes (%.1f×)", bits2Bytes, float64(origBytes)/float64(bits2Bytes))
	t.Logf("  K+V pair: %d bytes (%.1f× vs %d)", bits4Bytes+bits2Bytes,
		float64(2*origBytes)/float64(bits4Bytes+bits2Bytes), 2*origBytes)
}

func TestTurboQuantProtectedLayers(t *testing.T) {
	var nilTQ *TurboQuantState
	if nilTQ.IsProtectedLayer(0) {
		t.Fatal("nil TurboQuantState reported protected layer")
	}
	tq := NewTurboQuantState(128, 28, DefaultTurboQuantConfig())

	// First 2 and last 2 should be protected
	for _, l := range []int{0, 1, 26, 27} {
		if !tq.IsProtectedLayer(l) {
			t.Errorf("layer %d should be protected", l)
		}
	}
	for _, l := range []int{-1, 5, 14, 20} {
		if tq.IsProtectedLayer(l) {
			t.Errorf("layer %d should NOT be protected", l)
		}
	}
}

func TestTurboQuantOrthogonality(t *testing.T) {
	dim := 64
	tq := NewTurboQuantState(dim, 28, DefaultTurboQuantConfig())

	// Check R @ R^T ≈ I
	maxOffDiag := float64(0)
	minOnDiag := float64(2)
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			var dot float64
			for k := 0; k < dim; k++ {
				dot += float64(tq.RotationK[i*dim+k]) * float64(tq.RotationK[j*dim+k])
			}
			if i == j {
				if dot < minOnDiag {
					minOnDiag = dot
				}
			} else {
				if math.Abs(dot) > maxOffDiag {
					maxOffDiag = math.Abs(dot)
				}
			}
		}
	}
	t.Logf("Orthogonality check: minOnDiag=%.6f maxOffDiag=%.6g", minOnDiag, maxOffDiag)
	if maxOffDiag > 1e-5 || minOnDiag < 0.999 {
		t.Errorf("rotation matrix not orthogonal: minOnDiag=%.6f maxOffDiag=%.6g", minOnDiag, maxOffDiag)
	}
}

func TestCompressedKVCacheMalformedLayoutGuards(t *testing.T) {
	tq := NewTurboQuantState(4, 1, TurboQuantConfig{KeyBits: 4, ValueBits: 2, ResidualWindow: 0})
	bad := NewCompressedKVCache(4, 3, 2, tq, false)
	bad.Append([]float32{1, 2, 3, 4}, []float32{5, 6, 7, 8})
	if bad.CompressedCount() != 0 {
		t.Fatalf("inconsistent head layout compressed entries: %d", bad.CompressedCount())
	}

	c := NewCompressedKVCache(4, 1, 4, tq, false)
	c.FullK = []float32{1, 2, 3, 4, 9, 9, 9, 9}
	c.FullV = []float32{5, 6, 7, 8, 9, 9, 9, 9}
	c.seqLen = 1
	if got := c.GetK(); len(got) != 4 {
		t.Fatalf("GetK did not clamp full cache to seqLen: len=%d", len(got))
	}
	c.CompressedK = []compressedEntry{{Packed: []byte{1}, HeadVMin: nil, HeadScale: nil}}
	c.CompressedV = []compressedEntry{{Packed: []byte{1}, HeadVMin: nil, HeadScale: nil}}
	if got := c.GetK(); len(got) != len(c.FullK) {
		t.Fatalf("malformed compressed K should fall back to FullK, len=%d", len(got))
	}
	if got := c.GetV(); len(got) != len(c.FullV) {
		t.Fatalf("malformed compressed V should fall back to FullV, len=%d", len(got))
	}
}

func TestTurboQuantOverflowGuards(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	tooLarge := maxInt/2 + 1
	if got := squareSize(tooLarge); got != -1 {
		t.Fatalf("squareSize overflow=%d, want -1", got)
	}
	if got := packedByteLen(maxInt, 8); got != -1 {
		t.Fatalf("packedByteLen overflow=%d, want -1", got)
	}
	if got := randomOrthogonal(tooLarge, nil); got != nil {
		t.Fatalf("randomOrthogonal malformed returned len=%d", len(got))
	}
	state := NewTurboQuantState(tooLarge, 1, DefaultTurboQuantConfig())
	if state.HeadDim != 0 || len(state.RotationK) != 0 || len(state.RotationV) != 0 {
		t.Fatalf("overflowing headDim not sanitized: headDim=%d rk=%d rv=%d", state.HeadDim, len(state.RotationK), len(state.RotationV))
	}
	packed := packIndices(make([]byte, 4), 4)
	if len(packed) != 2 {
		t.Fatalf("packIndices valid len=%d want 2", len(packed))
	}
}
