package simd

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestCQMatrixMulMatchesDenseOracle(t *testing.T) {
	shapes := []struct {
		rows  int
		cols  int
		batch int
	}{
		{1, 1, 1},
		{2, 17, 3},
		{3, 127, 2},
		{4, 128, 1},
		{5, 129, 2},
		{6, 255, 3},
		{7, 257, 2},
	}
	for _, bits := range []int{1, 2, 3, 4, 5} {
		for _, shape := range shapes {
			t.Run(testCQName(bits, shape.rows, shape.cols, shape.batch), func(t *testing.T) {
				rng := rand.New(rand.NewSource(int64(bits*1_000_000 + shape.rows*10_000 + shape.cols*10 + shape.batch)))
				codebook := testCQCodebook()
				blob := makeRandomCQBlob(shape.rows, shape.cols, bits, rng)
				m, err := NewCQMatrix(shape.rows, shape.cols, bits, blob, codebook)
				if err != nil {
					t.Fatalf("NewCQMatrix: %v", err)
				}
				if got, want := m.Rows(), shape.rows; got != want {
					t.Fatalf("Rows=%d want %d", got, want)
				}
				if got, want := m.Cols(), shape.cols; got != want {
					t.Fatalf("Cols=%d want %d", got, want)
				}
				extra := int64(0)
				if bits == 2 || bits == 4 {
					extra = 4096
				}
				if got, want := m.Bytes(), int64(len(blob)+len(codebook)*4)+extra; got != want {
					t.Fatalf("Bytes=%d want %d", got, want)
				}

				input := make([]float32, shape.batch*shape.cols)
				for i := range input {
					input[i] = (rng.Float32()*2 - 1) * 0.25
				}
				dense := decodeCQDenseOracle(shape.rows, shape.cols, bits, blob, codebook)
				want := denseMulOracle(dense, shape.rows, shape.cols, input, shape.batch)
				got := make([]float32, shape.batch*shape.rows)
				if !m.Mul(got, input, shape.batch) {
					t.Fatal("Mul rejected valid inputs")
				}
				for i := range want {
					if diff := math.Abs(float64(got[i] - want[i])); diff > 5e-4 {
						t.Fatalf("out[%d]=%g want %g diff=%g", i, got[i], want[i], diff)
					}
				}
			})
		}
	}
}

func TestNewCQMatrixRejectsMalformed(t *testing.T) {
	codebook := testCQCodebook()
	if _, err := NewCQMatrix(0, 1, 4, nil, codebook); err == nil {
		t.Fatal("accepted zero rows")
	}
	if _, err := NewCQMatrix(1, 1, 6, nil, codebook); err == nil {
		t.Fatal("accepted unsupported bits")
	}
	if _, err := NewCQMatrix(2049, 65536, 1, nil, codebook); err == nil {
		t.Fatal("accepted decoded geometry > 512 MiB")
	}
	if _, err := NewCQMatrix(1, 128, 4, make([]byte, 1), codebook[:27]); err == nil {
		t.Fatal("accepted short codebook")
	}
	badCodebook := append([]float32(nil), codebook...)
	badCodebook[5] = badCodebook[4]
	if _, err := NewCQMatrix(1, 128, 4, make([]byte, 1), badCodebook); err == nil {
		t.Fatal("accepted unsorted codebook segment")
	}

	blob4 := makeRandomCQBlob(2, 129, 4, rand.New(rand.NewSource(1)))
	if _, err := NewCQMatrix(2, 129, 4, blob4[:len(blob4)-1], codebook); err == nil {
		t.Fatal("accepted short blob")
	}
	layout4 := testCQLayout(2, 129, 4)
	badNormNeg := append([]byte(nil), blob4...)
	binary.LittleEndian.PutUint16(badNormNeg[layout4.packedBytes:], 0xbc00)
	if _, err := NewCQMatrix(2, 129, 4, badNormNeg, codebook); err == nil {
		t.Fatal("accepted negative norm")
	}
	badNormInf := append([]byte(nil), blob4...)
	binary.LittleEndian.PutUint16(badNormInf[layout4.packedBytes:], 0x7c00)
	if _, err := NewCQMatrix(2, 129, 4, badNormInf, codebook); err == nil {
		t.Fatal("accepted non-finite norm")
	}

	blob5 := makeRandomCQBlob(1, 128, 5, rand.New(rand.NewSource(2)))
	blob5[0] = (blob5[0] &^ 0x3) | 0x2
	if _, err := NewCQMatrix(1, 128, 5, blob5, codebook); err == nil {
		t.Fatal("accepted invalid ternary crumb")
	}
}

func TestCQMatrixMulRejectsAliasAndNonFiniteInput(t *testing.T) {
	codebook := testCQCodebook()
	blob := makeRandomCQBlob(3, 5, 4, rand.New(rand.NewSource(3)))
	m, err := NewCQMatrix(3, 5, 4, blob, codebook)
	if err != nil {
		t.Fatal(err)
	}

	backing := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	input := backing[:5]
	dst := backing[2:5]
	beforeAlias := append([]float32(nil), dst...)
	if m.Mul(dst, input, 1) {
		t.Fatal("accepted overlapping dst/input")
	}
	for i := range dst {
		if math.Float32bits(dst[i]) != math.Float32bits(beforeAlias[i]) {
			t.Fatalf("dst changed on alias rejection: %v want %v", dst, beforeAlias)
		}
	}

	dst = []float32{9, 8, 7}
	beforeFinite := append([]float32(nil), dst...)
	badInput := []float32{1, 2, float32(math.NaN()), 4, 5}
	if m.Mul(dst, badInput, 1) {
		t.Fatal("accepted non-finite input")
	}
	for i := range dst {
		if math.Float32bits(dst[i]) != math.Float32bits(beforeFinite[i]) {
			t.Fatalf("dst changed on non-finite input: %v want %v", dst, beforeFinite)
		}
	}

	if m.Mul(nil, input, 1) {
		t.Fatal("accepted nil dst")
	}
	if m.Mul(make([]float32, 2), input, 1) {
		t.Fatal("accepted short dst")
	}
	if m.Mul(make([]float32, 3), input[:4], 1) {
		t.Fatal("accepted short input")
	}
	if m.Mul(make([]float32, 3), input, 0) {
		t.Fatal("accepted zero batch")
	}
}

func TestCQMatrixOwnsInputData(t *testing.T) {
	codebook := testCQCodebook()
	blob := makeRandomCQBlob(4, 129, 3, rand.New(rand.NewSource(4)))
	m, err := NewCQMatrix(4, 129, 3, blob, codebook)
	if err != nil {
		t.Fatal(err)
	}
	input := make([]float32, 129)
	for i := range input {
		input[i] = float32(i%7-3) * 0.125
	}
	want := make([]float32, 4)
	if !m.Mul(want, input, 1) {
		t.Fatal("Mul rejected valid baseline input")
	}

	for i := range blob {
		blob[i] ^= byte(i*37 + 11)
	}
	for i := range codebook {
		codebook[i] = float32(i) * 100
	}
	got := make([]float32, 4)
	if !m.Mul(got, input, 1) {
		t.Fatal("Mul rejected after caller mutation")
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("owned copy mismatch at %d: got %g want %g", i, got[i], want[i])
		}
	}
}

func TestCQMatrixConcurrentMul(t *testing.T) {
	codebook := testCQCodebook()
	blob := makeRandomCQBlob(8, 257, 4, rand.New(rand.NewSource(5)))
	m, err := NewCQMatrix(8, 257, 4, blob, codebook)
	if err != nil {
		t.Fatal(err)
	}
	input := make([]float32, 2*257)
	for i := range input {
		input[i] = float32((i%17)-8) * 0.03125
	}
	want := make([]float32, 16)
	if !m.Mul(want, input, 2) {
		t.Fatal("baseline Mul rejected valid input")
	}

	var wg sync.WaitGroup
	errCh := make(chan string, 16)
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 25; i++ {
				got := make([]float32, len(want))
				if !m.Mul(got, input, 2) {
					errCh <- "Mul rejected valid concurrent input"
					return
				}
				for j := range got {
					if math.Float32bits(got[j]) != math.Float32bits(want[j]) {
						errCh <- "concurrent result mismatch"
						return
					}
				}
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}
}

func BenchmarkCQMatrixMul576x768Batch1(b *testing.B) {
	codebook := testCQCodebook()
	blob := makeRandomCQBlob(576, 768, 4, rand.New(rand.NewSource(6)))
	m, err := NewCQMatrix(576, 768, 4, blob, codebook)
	if err != nil {
		b.Fatal(err)
	}
	input := make([]float32, 768)
	for i := range input {
		input[i] = float32((i%29)-14) * 0.015625
	}
	dst := make([]float32, 576)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !m.Mul(dst, input, 1) {
			b.Fatal("Mul rejected benchmark input")
		}
	}
}

func testCQName(bits, rows, cols, batch int) string {
	return fmt.Sprintf("bits%d_rows%d_cols%d_batch%d", bits, rows, cols, batch)
}

type testCQLayoutInfo struct {
	paddedCols     int
	groups         int
	packedPerGroup int
	packedPerRow   int
	packedBytes    int
	totalBytes     int
}

func testCQLayout(rows, cols, bits int) testCQLayoutInfo {
	groups := (cols + 127) / 128
	paddedCols := groups * 128
	packedPerGroup := 0
	switch bits {
	case 5:
		packedPerGroup = 32
	default:
		packedPerGroup = 128 * bits / 8
	}
	packedPerRow := groups * packedPerGroup
	packedBytes := rows * packedPerRow
	return testCQLayoutInfo{
		paddedCols:     paddedCols,
		groups:         groups,
		packedPerGroup: packedPerGroup,
		packedPerRow:   packedPerRow,
		packedBytes:    packedBytes,
		totalBytes:     packedBytes + rows*groups*2,
	}
}

func testCQCodebook() []float32 {
	return []float32{
		-0.90, -0.30, 0.20, 0.95,
		-0.98, -0.70, -0.42, -0.14, 0.14, 0.42, 0.70, 0.98,
		-1.05, -0.91, -0.77, -0.63, -0.49, -0.35, -0.21, -0.07,
		0.07, 0.21, 0.35, 0.49, 0.63, 0.77, 0.91, 1.05,
	}
}

func makeRandomCQBlob(rows, cols, bits int, rng *rand.Rand) []byte {
	layout := testCQLayout(rows, cols, bits)
	blob := make([]byte, layout.totalBytes)
	packed := blob[:layout.packedBytes]
	switch bits {
	case 5:
		for i := range packed {
			var v byte
			for j := 0; j < 4; j++ {
				crumbs := [...]byte{0, 1, 3}
				v |= crumbs[rng.Intn(len(crumbs))] << (2 * j)
			}
			packed[i] = v
		}
	default:
		rng.Read(packed)
	}
	for i := 0; i < rows*layout.groups; i++ {
		norm := float32(rng.Float64()*1.75 + 0.05)
		if i%7 == 0 {
			norm = 0
		}
		binary.LittleEndian.PutUint16(blob[layout.packedBytes+i*2:], half.F32ToF16Even(norm))
	}
	return blob
}

func decodeCQDenseOracle(rows, cols, bits int, blob []byte, codebook []float32) []float32 {
	layout := testCQLayout(rows, cols, bits)
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		rowPacked := blob[r*layout.packedPerRow : (r+1)*layout.packedPerRow]
		rowNorms := blob[layout.packedBytes+r*layout.groups*2 : layout.packedBytes+(r+1)*layout.groups*2]
		for g := 0; g < layout.groups; g++ {
			var work [128]float32
			norm := half.F16ToF32(binary.LittleEndian.Uint16(rowNorms[g*2:]))
			groupPacked := rowPacked[g*layout.packedPerGroup : (g+1)*layout.packedPerGroup]
			switch bits {
			case 1:
				oracleUnpackBinary(&work, groupPacked)
			case 2:
				oracleUnpackCodebook(&work, groupPacked, 2, codebook[:4])
			case 3:
				oracleUnpackCodebook(&work, groupPacked, 3, codebook[4:12])
			case 4:
				oracleUnpackCodebook(&work, groupPacked, 4, codebook[12:28])
			case 5:
				oracleUnpackTernary(&work, groupPacked)
			}
			for i := range work {
				work[i] *= norm
			}
			oracleWalsh128(work[:])
			base := g * 128
			width := cols - base
			if width > 128 {
				width = 128
			}
			copy(out[r*cols+base:r*cols+base+width], work[:width])
		}
	}
	return out
}

func denseMulOracle(weights []float32, rows, cols int, input []float32, batch int) []float32 {
	out := make([]float32, batch*rows)
	for b := 0; b < batch; b++ {
		for r := 0; r < rows; r++ {
			sum := 0.0
			row := weights[r*cols : (r+1)*cols]
			x := input[b*cols : (b+1)*cols]
			for c := 0; c < cols; c++ {
				sum += float64(row[c]) * float64(x[c])
			}
			out[b*rows+r] = float32(sum)
		}
	}
	return out
}

func oracleUnpackBinary(dst *[128]float32, packed []byte) {
	level := float32(math.Sqrt(2/math.Pi) / math.Sqrt(128))
	for i, b := range packed {
		base := i * 8
		for j := 0; j < 8; j++ {
			if (b>>j)&1 != 0 {
				dst[base+j] = level
			} else {
				dst[base+j] = -level
			}
		}
	}
}

func oracleUnpackCodebook(dst *[128]float32, packed []byte, bits int, codebook []float32) {
	mask := uint32((1 << bits) - 1)
	for chunk := 0; chunk < 16; chunk++ {
		base := chunk * bits
		word := uint32(0)
		for b := 0; b < bits; b++ {
			word |= uint32(packed[base+b]) << (8 * b)
		}
		for i := 0; i < 8; i++ {
			dst[chunk*8+i] = codebook[int((word>>(i*bits))&mask)]
		}
	}
}

func oracleUnpackTernary(dst *[128]float32, packed []byte) {
	level := float32(1.2240064 / math.Sqrt(128))
	for i, b := range packed {
		base := i * 4
		for j := 0; j < 4; j++ {
			switch (b >> (2 * j)) & 0x3 {
			case 3:
				dst[base+j] = -level
			case 1:
				dst[base+j] = level
			default:
				dst[base+j] = 0
			}
		}
	}
}

func oracleWalsh128(x []float32) {
	scale := float32(1 / math.Sqrt(128))
	for step := 1; step < 128; step <<= 1 {
		block := step << 1
		for base := 0; base < 128; base += block {
			for i := 0; i < step; i++ {
				a := x[base+i]
				b := x[base+step+i]
				x[base+i] = a + b
				x[base+step+i] = a - b
			}
		}
	}
	for i := range x {
		x[i] *= scale
	}
}

func TestCQZeroAndScratchAdmission(t *testing.T) {
	var zero CQMatrix
	dst := []float32{9}
	if zero.Mul(dst, []float32{1}, 1) || dst[0] != 9 {
		t.Fatal("zero matrix mutated output")
	}
	if _, err := NewCQMatrix(1, 1<<25, 4, nil, testCQCodebook()); err == nil {
		t.Fatal("oversized per-row scratch admitted")
	}
}
