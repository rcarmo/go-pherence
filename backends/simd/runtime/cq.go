package simd

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/internal/checked"
)

const (
	cqGroup128       = 128
	cqCodebookLen    = 28
	cqTernaryBits    = 5
	cqMaxBlobBytes   = 512 << 20
	cqMaxDecodedByte = 512 << 20
	cqScratchMaxByte = 64 << 20
)

var (
	cqWalshScale128 = float32(1 / math.Sqrt(float64(cqGroup128)))
	cqBinaryLevel   = float32(math.Sqrt(2/math.Pi) / math.Sqrt(float64(cqGroup128)))
	cqTernaryLevel  = float32(1.2240064 / math.Sqrt(float64(cqGroup128)))
)

// CQMatrix is an immutable Needle .cact packed CQ matrix with fixed group-128
// Walsh geometry. The packed payload is row-major by output row, with all row
// norms appended after the packed index stream.
type CQMatrix struct {
	rows           int
	cols           int
	bits           int
	paddedCols     int
	groups         int
	packedPerGroup int
	packedPerRow   int
	blob           []byte
	packed         []byte
	norms          []byte
	codebook       []float32
	bytes          int64
	lookup         [][4]float32 // byte expansion for measured 2-/4-bit hotspots
}

// NewCQMatrix validates and owns a packed CQ matrix payload.
func NewCQMatrix(rows, cols, bits int, blob []byte, codebook []float32) (*CQMatrix, error) {
	layout, err := cqLayout(rows, cols, bits)
	if err != nil {
		return nil, err
	}
	if err := validateCQCodebook(codebook); err != nil {
		return nil, err
	}
	if len(blob) > cqMaxBlobBytes {
		return nil, fmt.Errorf("simd: CQ blob exceeds 512 MiB")
	}
	if len(blob) != layout.totalBytes {
		return nil, fmt.Errorf("simd: CQ blob size %d want %d", len(blob), layout.totalBytes)
	}

	packed := blob[:layout.packedBytes]
	norms := blob[layout.packedBytes:]
	if err := validateCQNorms(norms); err != nil {
		return nil, err
	}
	if bits == cqTernaryBits {
		if err := validateCQTernaryPacked(packed); err != nil {
			return nil, err
		}
	}

	blobOwned := append([]byte(nil), blob...)
	packed = blobOwned[:layout.packedBytes]
	norms = blobOwned[layout.packedBytes:]
	cbOwned := append([]float32(nil), codebook...)
	var lookup [][4]float32
	if bits == 2 || bits == 4 {
		lookup = make([][4]float32, 256)
		for i := range lookup {
			if bits == 2 {
				for j := 0; j < 4; j++ {
					lookup[i][j] = cbOwned[(i>>(2*j))&3]
				}
			} else {
				lookup[i][0] = cbOwned[12+(i&15)]
				lookup[i][1] = cbOwned[12+(i>>4)]
			}
		}
	}
	return &CQMatrix{
		rows:           rows,
		cols:           cols,
		bits:           bits,
		paddedCols:     layout.paddedCols,
		groups:         layout.groups,
		packedPerGroup: layout.packedPerGroup,
		packedPerRow:   layout.packedPerRow,
		blob:           blobOwned,
		packed:         packed,
		norms:          norms,
		codebook:       cbOwned,
		bytes:          int64(len(blobOwned)) + int64(len(cbOwned))*4 + int64(len(lookup))*16,
		lookup:         lookup,
	}, nil
}

func (m *CQMatrix) Rows() int {
	if m == nil {
		return 0
	}
	return m.rows
}

func (m *CQMatrix) Cols() int {
	if m == nil {
		return 0
	}
	return m.cols
}

func (m *CQMatrix) Bytes() int64 {
	if m == nil {
		return 0
	}
	return m.bytes
}

// Mul overwrites dst[batch,rows] with input[batch,cols] * CQ^T.
func (m *CQMatrix) Mul(dst, input []float32, batch int) bool {
	if m == nil || m.rows <= 0 || m.cols <= 0 || batch <= 0 {
		return false
	}
	needDst, okDst := checked.MulInt(batch, m.rows)
	needIn, okIn := checked.MulInt(batch, m.cols)
	if !okDst || !okIn || len(dst) < needDst || len(input) < needIn {
		return false
	}
	dst = dst[:needDst]
	input = input[:needIn]
	if overlapFloat32(dst, input) {
		return false
	}
	for _, v := range input {
		if !cqFinite32(v) {
			return false
		}
	}

	if scratchElems, ok := checked.MulInt(batch, m.paddedCols); ok {
		if scratchBytes, ok := checked.MulInt(scratchElems, 4); ok && scratchBytes <= cqScratchMaxByte {
			scratch := make([]float32, scratchElems)
			for b := 0; b < batch; b++ {
				m.transformInput(scratch[b*m.paddedCols:(b+1)*m.paddedCols], input[b*m.cols:(b+1)*m.cols])
			}
			for b := 0; b < batch; b++ {
				m.mulOne(dst[b*m.rows:(b+1)*m.rows], scratch[b*m.paddedCols:(b+1)*m.paddedCols])
			}
			return true
		}
	}

	scratch := make([]float32, m.paddedCols)
	for b := 0; b < batch; b++ {
		m.transformInput(scratch, input[b*m.cols:(b+1)*m.cols])
		m.mulOne(dst[b*m.rows:(b+1)*m.rows], scratch)
	}
	return true
}

type cqMatrixLayout struct {
	paddedCols     int
	groups         int
	packedPerGroup int
	packedPerRow   int
	packedBytes    int
	totalBytes     int
}

func cqLayout(rows, cols, bits int) (cqMatrixLayout, error) {
	if rows <= 0 || cols <= 0 {
		return cqMatrixLayout{}, fmt.Errorf("simd: invalid CQ shape [%d %d]", rows, cols)
	}
	switch bits {
	case 1, 2, 3, 4, cqTernaryBits:
	default:
		return cqMatrixLayout{}, fmt.Errorf("simd: unsupported CQ bits %d", bits)
	}
	decodedElems, ok := checked.MulInt(rows, cols)
	if !ok {
		return cqMatrixLayout{}, fmt.Errorf("simd: CQ decoded geometry overflows")
	}
	if decodedElems > cqMaxDecodedByte/4 {
		return cqMatrixLayout{}, fmt.Errorf("simd: CQ decoded geometry exceeds 512 MiB")
	}
	colsPlus, ok := checked.AddInt(cols, cqGroup128-1)
	if !ok {
		return cqMatrixLayout{}, fmt.Errorf("simd: CQ width overflows")
	}
	groups := colsPlus / cqGroup128
	paddedCols, ok := checked.MulInt(groups, cqGroup128)
	if !ok {
		return cqMatrixLayout{}, fmt.Errorf("simd: CQ padded width overflows")
	}
	if paddedCols > cqScratchMaxByte/4 {
		return cqMatrixLayout{}, fmt.Errorf("simd: CQ row scratch exceeds 64 MiB")
	}
	packedPerGroup := 0
	switch bits {
	case cqTernaryBits:
		packedPerGroup = cqGroup128 / 4
	default:
		packedPerGroup = cqGroup128 * bits / 8
	}
	packedPerRow, ok := checked.MulInt(groups, packedPerGroup)
	if !ok {
		return cqMatrixLayout{}, fmt.Errorf("simd: CQ row byte size overflows")
	}
	packedBytes, ok := checked.MulInt(rows, packedPerRow)
	if !ok {
		return cqMatrixLayout{}, fmt.Errorf("simd: CQ packed byte size overflows")
	}
	normCount, ok := checked.MulInt(rows, groups)
	if !ok {
		return cqMatrixLayout{}, fmt.Errorf("simd: CQ norm count overflows")
	}
	normBytes, ok := checked.MulInt(normCount, 2)
	if !ok {
		return cqMatrixLayout{}, fmt.Errorf("simd: CQ norm byte size overflows")
	}
	totalBytes, ok := checked.AddInt(packedBytes, normBytes)
	if !ok {
		return cqMatrixLayout{}, fmt.Errorf("simd: CQ blob size overflows")
	}
	if totalBytes > cqMaxBlobBytes {
		return cqMatrixLayout{}, fmt.Errorf("simd: CQ blob exceeds 512 MiB")
	}
	return cqMatrixLayout{
		paddedCols:     paddedCols,
		groups:         groups,
		packedPerGroup: packedPerGroup,
		packedPerRow:   packedPerRow,
		packedBytes:    packedBytes,
		totalBytes:     totalBytes,
	}, nil
}

func validateCQCodebook(codebook []float32) error {
	if len(codebook) != cqCodebookLen {
		return fmt.Errorf("simd: CQ codebook length %d want %d", len(codebook), cqCodebookLen)
	}
	if err := validateCQCodebookSegment(codebook, 0, 4, 2); err != nil {
		return err
	}
	if err := validateCQCodebookSegment(codebook, 4, 12, 3); err != nil {
		return err
	}
	if err := validateCQCodebookSegment(codebook, 12, 28, 4); err != nil {
		return err
	}
	return nil
}

func validateCQCodebookSegment(codebook []float32, start, end, bits int) error {
	seg := codebook[start:end]
	for i, v := range seg {
		if !cqFinite32(v) {
			return fmt.Errorf("simd: CQ codebook[%d] for %d-bit segment is non-finite", start+i, bits)
		}
		if i > 0 && !(seg[i-1] < v) {
			return fmt.Errorf("simd: CQ %d-bit codebook is not strictly increasing", bits)
		}
	}
	return nil
}

func validateCQNorms(norms []byte) error {
	for i := 0; i < len(norms); i += 2 {
		norm := half.F16ToF32(binary.LittleEndian.Uint16(norms[i:]))
		if !cqFinite32(norm) || norm < 0 {
			return fmt.Errorf("simd: CQ norm is invalid")
		}
	}
	return nil
}

func validateCQTernaryPacked(packed []byte) error {
	for _, b := range packed {
		for shift := 0; shift < 8; shift += 2 {
			if ((b >> shift) & 0x3) == 2 {
				return fmt.Errorf("simd: invalid CQ ternary crumb 2")
			}
		}
	}
	return nil
}

func (m *CQMatrix) transformInput(dst, src []float32) {
	clear(dst)
	copy(dst, src)
	for base := 0; base < m.paddedCols; base += cqGroup128 {
		cqWalsh128(dst[base : base+cqGroup128])
	}
}

func (m *CQMatrix) mulOne(dst, transformed []float32) {
	var work [cqGroup128]float32
	for row := 0; row < m.rows; row++ {
		rowPacked := m.packed[row*m.packedPerRow : (row+1)*m.packedPerRow]
		rowNorms := m.norms[row*m.groups*2 : (row+1)*m.groups*2]
		sum := float32(0)
		for g := 0; g < m.groups; g++ {
			norm := half.F16ToF32(binary.LittleEndian.Uint16(rowNorms[g*2:]))
			if norm == 0 {
				continue
			}
			groupPacked := rowPacked[g*m.packedPerGroup : (g+1)*m.packedPerGroup]
			switch m.bits {
			case 1:
				unpackCQBinary(&work, groupPacked)
			case 2:
				for i, b := range groupPacked {
					copy(work[i*4:i*4+4], m.lookup[b][:])
				}
			case 3:
				unpackCQCodebook(&work, groupPacked, 3, m.codebook[4:12])
			case 4:
				for i, b := range groupPacked {
					pair := m.lookup[b]
					work[2*i], work[2*i+1] = pair[0], pair[1]
				}
			case cqTernaryBits:
				unpackCQTernary(&work, groupPacked)
			}
			sum += Sdot(work[:], transformed[g*cqGroup128:(g+1)*cqGroup128]) * norm
		}
		dst[row] = sum
	}
}

func unpackCQBinary(dst *[cqGroup128]float32, packed []byte) {
	for i, b := range packed {
		base := i * 8
		for j := 0; j < 8; j++ {
			if (b>>j)&1 != 0 {
				dst[base+j] = cqBinaryLevel
			} else {
				dst[base+j] = -cqBinaryLevel
			}
		}
	}
}

func unpackCQCodebook(dst *[cqGroup128]float32, packed []byte, bits int, codebook []float32) {
	mask := uint32((1 << bits) - 1)
	for chunk := 0; chunk < cqGroup128/8; chunk++ {
		base := chunk * bits
		word := uint32(0)
		for b := 0; b < bits; b++ {
			word |= uint32(packed[base+b]) << (8 * b)
		}
		outBase := chunk * 8
		for i := 0; i < 8; i++ {
			dst[outBase+i] = codebook[int((word>>(i*bits))&mask)]
		}
	}
}

func unpackCQTernary(dst *[cqGroup128]float32, packed []byte) {
	for i, b := range packed {
		base := i * 4
		for j := 0; j < 4; j++ {
			switch (b >> (2 * j)) & 0x3 {
			case 3:
				dst[base+j] = -cqTernaryLevel
			case 0:
				dst[base+j] = 0
			case 1:
				dst[base+j] = cqTernaryLevel
			default:
				dst[base+j] = 0
			}
		}
	}
}

func cqWalsh128(x []float32) {
	for step := 1; step < cqGroup128; step <<= 1 {
		block := step << 1
		for base := 0; base < cqGroup128; base += block {
			for i := 0; i < step; i++ {
				a := x[base+i]
				b := x[base+step+i]
				x[base+i] = a + b
				x[base+step+i] = a - b
			}
		}
	}
	for i := range x {
		x[i] *= cqWalshScale128
	}
}

func cqFinite32(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

func overlapFloat32(a, b []float32) bool {
	return !float32SlicesDisjoint(a, b)
}
