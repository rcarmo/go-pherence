package kv

import (
	"math"
	"testing"
)

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
