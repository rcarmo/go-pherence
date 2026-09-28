package nvidia

import (
	"encoding/binary"
	"fmt"
	"unsafe"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/internal/checked"
)

var (
	fnQ4KGemv                 CUfunction
	fnQ4KGemvBatch            CUfunction
	fnQ4KGemmBatch8           CUfunction
	fnQuantizeQ8RowsSum       CUfunction
	fnQ4KQ8Batch4             CUfunction
	fnQ4KQ8Batch8             CUfunction
	fnQ4KQ8Batch16            CUfunction
	fnQ4CoalescedQ8Batch8     CUfunction
	fnQ4CoalescedQ8           CUfunction
	fnQ4CoalescedF32          CUfunction
	fnQ4CoalescedMMQ8         CUfunction
	fnQ4CoalescedMMQJ16       CUfunction
	fnQ4PairMMQJ16            CUfunction
	fnQ4UpstreamMMQJ8         CUfunction
	fnQ4UpstreamMMQJ16        CUfunction
	fnQ4UpstreamMMQJ24        CUfunction
	fnQ4UpstreamMMQJ32        CUfunction
	fnQ4UpstreamMMQJ64        CUfunction
	fnQuantizeQ81MMQ          CUfunction
	fnQ4RawF32                CUfunction
	fnQ4KGateUpGELU           CUfunction
	fnQ4KGateUpGELUByWork     CUfunction
	fnQ4KGateUpGELUByWorkPtrs CUfunction
	fnQ4KGateUpByWorkPtrs     CUfunction
)

type GPUQ4KMatrix struct {
	Q         *Buffer // GGUF packed bytes, or coalesced [outDim, inDim/32, 16]
	Scales    *Buffer // [outDim, inDim/32]
	Mins      *Buffer // [outDim, inDim/32]
	InDim     int
	OutDim    int
	coalesced bool
	raw       bool
}

type GPUQ4KPointerTable struct {
	QPtrs     *Buffer // uint64 device pointers, one per active expert
	ScalePtrs *Buffer // uint64 device pointers, one per active expert
	MinPtrs   *Buffer // uint64 device pointers, one per active expert
	InDim     int
	OutDim    int
	Count     int
}

func UploadQ4KPointerTable(mats []*GPUQ4KMatrix) (*GPUQ4KPointerTable, error) {
	if len(mats) == 0 {
		return nil, fmt.Errorf("empty Q4_K pointer table")
	}
	inDim, outDim := mats[0].InDim, mats[0].OutDim
	qPtrs := make([]byte, len(mats)*8)
	sPtrs := make([]byte, len(mats)*8)
	mPtrs := make([]byte, len(mats)*8)
	for i, m := range mats {
		if m == nil || m.Q == nil || m.Scales == nil || m.Mins == nil || m.Q.Ptr == 0 || m.Scales.Ptr == 0 || m.Mins.Ptr == 0 || m.InDim != inDim || m.OutDim != outDim {
			return nil, fmt.Errorf("invalid Q4_K matrix %d for pointer table", i)
		}
		binary.LittleEndian.PutUint64(qPtrs[i*8:(i+1)*8], uint64(m.Q.Ptr))
		binary.LittleEndian.PutUint64(sPtrs[i*8:(i+1)*8], uint64(m.Scales.Ptr))
		binary.LittleEndian.PutUint64(mPtrs[i*8:(i+1)*8], uint64(m.Mins.Ptr))
	}
	qBuf, err := MallocBytes(len(qPtrs))
	if err != nil {
		return nil, err
	}
	if err := qBuf.UploadBytes(qPtrs); err != nil {
		qBuf.Free()
		return nil, err
	}
	sBuf, err := MallocBytes(len(sPtrs))
	if err != nil {
		qBuf.Free()
		return nil, err
	}
	if err := sBuf.UploadBytes(sPtrs); err != nil {
		qBuf.Free()
		sBuf.Free()
		return nil, err
	}
	mBuf, err := MallocBytes(len(mPtrs))
	if err != nil {
		qBuf.Free()
		sBuf.Free()
		return nil, err
	}
	if err := mBuf.UploadBytes(mPtrs); err != nil {
		qBuf.Free()
		sBuf.Free()
		mBuf.Free()
		return nil, err
	}
	return &GPUQ4KPointerTable{QPtrs: qBuf, ScalePtrs: sBuf, MinPtrs: mBuf, InDim: inDim, OutDim: outDim, Count: len(mats)}, nil
}

func (t *GPUQ4KPointerTable) Free() {
	if t == nil {
		return
	}
	if t.QPtrs != nil {
		t.QPtrs.Free()
		t.QPtrs = nil
	}
	if t.ScalePtrs != nil {
		t.ScalePtrs.Free()
		t.ScalePtrs = nil
	}
	if t.MinPtrs != nil {
		t.MinPtrs.Free()
		t.MinPtrs = nil
	}
}

func unpackQ4KMatrixRows(raw []byte, inDim, outDim int) ([]byte, []float32, []float32, error) {
	if inDim <= 0 || outDim <= 0 || inDim%256 != 0 {
		return nil, nil, nil, fmt.Errorf("invalid Q4_K dims in=%d out=%d", inDim, outDim)
	}
	blocks := inDim / 256
	rowBytes := blocks * 144
	needRaw, okRaw := checked.MulInt(rowBytes, outDim)
	qLen := outDim * blocks * 128
	sLen := outDim * blocks * 8
	if !okRaw || len(raw) < needRaw {
		return nil, nil, nil, fmt.Errorf("invalid Q4_K raw len=%d need=%d", len(raw), needRaw)
	}
	q := make([]byte, qLen)
	scales := make([]float32, sLen)
	mins := make([]float32, sLen)
	for r := 0; r < outDim; r++ {
		row := raw[r*rowBytes : (r+1)*rowBytes]
		for b := 0; b < blocks; b++ {
			blk := row[b*144:]
			d := half.F16ToF32(binary.LittleEndian.Uint16(blk[0:2]))
			dmin := half.F16ToF32(binary.LittleEndian.Uint16(blk[2:4]))
			sc := blk[4:16]
			baseS := (r*blocks + b) * 8
			for j := 0; j < 4; j++ {
				scales[baseS+j] = float32(sc[j]&63) * d
				mins[baseS+j] = float32(sc[j+4]&63) * dmin
			}
			for j := 4; j < 8; j++ {
				k := j - 4
				scales[baseS+j] = float32((sc[j+4]&0xF)|((sc[k]>>6)<<4)) * d
				mins[baseS+j] = float32((sc[j+4]>>4)|((sc[k+4]>>6)<<4)) * dmin
			}
			copy(q[(r*blocks+b)*128:(r*blocks+b+1)*128], blk[16:144])
		}
	}
	return q, scales, mins, nil
}

func UploadQ4KMatrixRows(raw []byte, inDim, outDim int) (*GPUQ4KMatrix, error) {
	return uploadQ4KMatrixRows(raw, inDim, outDim, false)
}

// UploadQ4KMatrixRowsCoalesced preserves the Q4_K bit width while arranging
// each 32-value group contiguously for the Go System One DP4A kernels.
func UploadQ4KMatrixRowsCoalesced(raw []byte, inDim, outDim int) (*GPUQ4KMatrix, error) {
	return uploadQ4KMatrixRows(raw, inDim, outDim, true)
}
func UploadQ4KMatrixRaw(raw []byte, inDim, outDim int) (*GPUQ4KMatrix, error) {
	if inDim <= 0 || outDim <= 0 || inDim%256 != 0 || len(raw) < outDim*(inDim/256)*144 {
		return nil, fmt.Errorf("invalid raw Q4_K matrix")
	}
	b, err := MallocBytes(len(raw))
	if err != nil {
		return nil, err
	}
	if err = b.UploadBytes(raw); err != nil {
		b.Free()
		return nil, err
	}
	return &GPUQ4KMatrix{Q: b, InDim: inDim, OutDim: outDim, raw: true}, nil
}
func uploadQ4KMatrixRows(raw []byte, inDim, outDim int, coalesced bool) (*GPUQ4KMatrix, error) {
	q, scales, mins, err := unpackQ4KMatrixRows(raw, inDim, outDim)
	if err != nil {
		return nil, err
	}
	if coalesced {
		q = coalesceQ4K(q, inDim, outDim)
	}
	qBuf, err := MallocBytes(len(q))
	if err != nil {
		return nil, err
	}
	if err := qBuf.UploadBytes(q); err != nil {
		qBuf.Free()
		return nil, err
	}
	sBuf, err := Malloc(len(scales))
	if err != nil {
		qBuf.Free()
		return nil, err
	}
	if err := sBuf.Upload(scales); err != nil {
		qBuf.Free()
		sBuf.Free()
		return nil, err
	}
	mBuf, err := Malloc(len(mins))
	if err != nil {
		qBuf.Free()
		sBuf.Free()
		return nil, err
	}
	if err := mBuf.Upload(mins); err != nil {
		qBuf.Free()
		sBuf.Free()
		mBuf.Free()
		return nil, err
	}
	return &GPUQ4KMatrix{Q: qBuf, Scales: sBuf, Mins: mBuf, InDim: inDim, OutDim: outDim, coalesced: coalesced}, nil
}

func coalesceQ4K(q []byte, inDim, outDim int) []byte {
	blocks := inDim / 256
	out := make([]byte, len(q))
	for r := 0; r < outDim; r++ {
		for b := 0; b < blocks; b++ {
			src := q[(r*blocks+b)*128:]
			for g := 0; g < 8; g++ {
				dst := out[(r*(inDim/32)+b*8+g)*16:]
				for i := 0; i < 32; i++ {
					v := src[(g/2)*32+i]
					if g&1 != 0 {
						v >>= 4
					} else {
						v &= 15
					}
					if i&1 == 0 {
						dst[i/2] = v
					} else {
						dst[i/2] |= v << 4
					}
				}
			}
		}
	}
	return out
}

func UploadQ4KMatrixRowsInto(m *GPUQ4KMatrix, raw []byte, inDim, outDim int) error {
	if m == nil || m.Q == nil || m.Scales == nil || m.Mins == nil || m.Q.Ptr == 0 || m.Scales.Ptr == 0 || m.Mins.Ptr == 0 || m.InDim != inDim || m.OutDim != outDim {
		return fmt.Errorf("invalid destination Q4_K matrix for in-place upload")
	}
	q, scales, mins, err := unpackQ4KMatrixRows(raw, inDim, outDim)
	if err != nil {
		return err
	}
	if m.Q.Size < len(q) || m.Scales.Size < len(scales)*4 || m.Mins.Size < len(mins)*4 {
		return fmt.Errorf("destination Q4_K matrix too small q=%d/%d scales=%d/%d mins=%d/%d", m.Q.Size, len(q), m.Scales.Size, len(scales)*4, m.Mins.Size, len(mins)*4)
	}
	if err := m.Q.UploadBytes(q); err != nil {
		return err
	}
	if err := m.Scales.Upload(scales); err != nil {
		return err
	}
	return m.Mins.Upload(mins)
}

func (m *GPUQ4KMatrix) Free() {
	if m == nil {
		return
	}
	if m.Q != nil {
		m.Q.Free()
		m.Q = nil
	}
	if m.Scales != nil {
		m.Scales.Free()
		m.Scales = nil
	}
	if m.Mins != nil {
		m.Mins.Free()
		m.Mins = nil
	}
}

func GateUpQ4KByWorkPtrsToBuffers(gateBuf, upBuf, xBuf, workExperts *Buffer, workLen, intermediate int, table *GPUQ4KPointerTable) error {
	if workLen <= 0 {
		return nil
	}
	if table == nil || table.QPtrs == nil || table.ScalePtrs == nil || table.MinPtrs == nil || table.Count <= 0 || table.InDim <= 0 || table.OutDim < intermediate*2 || xBuf == nil || gateBuf == nil || upBuf == nil || workExperts == nil || xBuf.Ptr == 0 || gateBuf.Ptr == 0 || upBuf.Ptr == 0 || workExperts.Ptr == 0 || table.QPtrs.Ptr == 0 || table.ScalePtrs.Ptr == 0 || table.MinPtrs.Ptr == 0 || xBuf.Size < workLen*table.InDim*4 || gateBuf.Size < workLen*intermediate*4 || upBuf.Size < workLen*intermediate*4 || workExperts.Size < workLen*4 || fnQ4KGateUpByWorkPtrs == 0 {
		return fmt.Errorf("invalid Q4_K gate/up by-work pointer-table buffers")
	}
	inDim := uint32(table.InDim)
	inter := uint32(intermediate)
	work := uint32(workLen)
	active := uint32(table.Count)
	args := []unsafe.Pointer{unsafe.Pointer(&xBuf.Ptr), unsafe.Pointer(&workExperts.Ptr), unsafe.Pointer(&table.QPtrs.Ptr), unsafe.Pointer(&table.ScalePtrs.Ptr), unsafe.Pointer(&table.MinPtrs.Ptr), unsafe.Pointer(&gateBuf.Ptr), unsafe.Pointer(&upBuf.Ptr), unsafe.Pointer(&inDim), unsafe.Pointer(&inter), unsafe.Pointer(&work), unsafe.Pointer(&active)}
	return LaunchKernel(fnQ4KGateUpByWorkPtrs, uint32(intermediate), uint32(workLen), 1, 256, 1, 1, 0, args...)
}

func GateUpGELUQ4KByWorkPtrsToBuffer(outBuf, xBuf, workExperts *Buffer, workLen, intermediate int, table *GPUQ4KPointerTable) error {
	if workLen <= 0 {
		return nil
	}
	if table == nil || table.QPtrs == nil || table.ScalePtrs == nil || table.MinPtrs == nil || table.Count <= 0 || table.InDim <= 0 || table.OutDim < intermediate*2 || xBuf == nil || outBuf == nil || workExperts == nil || xBuf.Ptr == 0 || outBuf.Ptr == 0 || workExperts.Ptr == 0 || table.QPtrs.Ptr == 0 || table.ScalePtrs.Ptr == 0 || table.MinPtrs.Ptr == 0 || xBuf.Size < workLen*table.InDim*4 || outBuf.Size < workLen*intermediate*4 || workExperts.Size < workLen*4 || fnQ4KGateUpGELUByWorkPtrs == 0 {
		return fmt.Errorf("invalid Q4_K gate/up GELU by-work pointer-table buffers")
	}
	inDim := uint32(table.InDim)
	inter := uint32(intermediate)
	work := uint32(workLen)
	active := uint32(table.Count)
	args := []unsafe.Pointer{unsafe.Pointer(&xBuf.Ptr), unsafe.Pointer(&workExperts.Ptr), unsafe.Pointer(&table.QPtrs.Ptr), unsafe.Pointer(&table.ScalePtrs.Ptr), unsafe.Pointer(&table.MinPtrs.Ptr), unsafe.Pointer(&outBuf.Ptr), unsafe.Pointer(&inDim), unsafe.Pointer(&inter), unsafe.Pointer(&work), unsafe.Pointer(&active)}
	return LaunchKernel(fnQ4KGateUpGELUByWorkPtrs, uint32(intermediate), uint32(workLen), 1, 256, 1, 1, 0, args...)
}

func GateUpGELUQ4KByWorkToBuffer(outBuf, xBuf, workExperts *Buffer, workLen, intermediate, activeExperts int, m *GPUQ4KMatrix) error {
	if workLen <= 0 {
		return nil
	}
	if m == nil || m.Q == nil || m.Scales == nil || m.Mins == nil || xBuf == nil || outBuf == nil || workExperts == nil || xBuf.Ptr == 0 || outBuf.Ptr == 0 || workExperts.Ptr == 0 || xBuf.Size < workLen*m.InDim*4 || outBuf.Size < workLen*intermediate*4 || workExperts.Size < workLen*4 || m.OutDim < activeExperts*intermediate*2 || fnQ4KGateUpGELUByWork == 0 {
		return fmt.Errorf("invalid Q4_K gate/up GELU by-work buffers")
	}
	inDim := uint32(m.InDim)
	inter := uint32(intermediate)
	work := uint32(workLen)
	active := uint32(activeExperts)
	args := []unsafe.Pointer{unsafe.Pointer(&xBuf.Ptr), unsafe.Pointer(&workExperts.Ptr), unsafe.Pointer(&m.Q.Ptr), unsafe.Pointer(&m.Scales.Ptr), unsafe.Pointer(&m.Mins.Ptr), unsafe.Pointer(&outBuf.Ptr), unsafe.Pointer(&inDim), unsafe.Pointer(&inter), unsafe.Pointer(&work), unsafe.Pointer(&active)}
	return LaunchKernel(fnQ4KGateUpGELUByWork, uint32(intermediate), uint32(workLen), 1, 256, 1, 1, 0, args...)
}

func GateUpGELUQ4KBatchToBuffer(outBuf *Buffer, xBuf *Buffer, batch, intermediate int, m *GPUQ4KMatrix) error {
	if batch <= 0 {
		return nil
	}
	if m == nil || m.Q == nil || m.Scales == nil || m.Mins == nil || xBuf == nil || outBuf == nil || xBuf.Ptr == 0 || outBuf.Ptr == 0 || xBuf.Size < batch*m.InDim*4 || outBuf.Size < batch*intermediate*4 || m.OutDim != intermediate*2 || fnQ4KGateUpGELU == 0 {
		return fmt.Errorf("invalid Q4_K gate/up GELU buffers")
	}
	inDim := uint32(m.InDim)
	inter := uint32(intermediate)
	batchU := uint32(batch)
	args := []unsafe.Pointer{unsafe.Pointer(&xBuf.Ptr), unsafe.Pointer(&m.Q.Ptr), unsafe.Pointer(&m.Scales.Ptr), unsafe.Pointer(&m.Mins.Ptr), unsafe.Pointer(&outBuf.Ptr), unsafe.Pointer(&inDim), unsafe.Pointer(&inter), unsafe.Pointer(&batchU)}
	return LaunchKernel(fnQ4KGateUpGELU, uint32(intermediate), uint32(batch), 1, 256, 1, 1, 0, args...)
}

func GemvQ4KBatchToBuffer(outBuf *Buffer, xBuf *Buffer, batch int, m *GPUQ4KMatrix) error {
	if m != nil && m.raw {
		if batch >= 4 {
			return gemmQ4RawUpstream(outBuf, xBuf, batch, m)
		}
		if fnQ4RawF32 != 0 {
			kk, nn, bb := uint32(m.InDim), uint32(m.OutDim), uint32(batch)
			return LaunchKernel(fnQ4RawF32, uint32((m.OutDim+3)/4), bb, 1, 128, 1, 1, 0, unsafe.Pointer(&xBuf.Ptr), unsafe.Pointer(&m.Q.Ptr), unsafe.Pointer(&outBuf.Ptr), unsafe.Pointer(&kk), unsafe.Pointer(&nn), unsafe.Pointer(&bb))
		}
	}
	if m != nil && m.coalesced {
		if batch >= 4 && fnQuantizeQ8RowsSum != 0 && fnQ4CoalescedQ8 != 0 {
			return GemmQ4CoalescedQ8ToBuffer(outBuf, xBuf, batch, m)
		}
		if fnQ4CoalescedF32 != 0 {
			return GemmQ4CoalescedF32ToBuffer(outBuf, xBuf, batch, m)
		}
	}
	if batch >= 4 && fnQuantizeQ8RowsSum != 0 && fnQ4KQ8Batch4 != 0 {
		return GemmQ4KQ8ToBuffer(outBuf, xBuf, batch, m)
	}
	if batch >= 4 && fnQ4KGemmBatch8 != 0 {
		return GemmQ4KBatch8ToBuffer(outBuf, xBuf, batch, m)
	}
	if batch <= 0 {
		return nil
	}
	if m == nil || m.Q == nil || m.Scales == nil || m.Mins == nil || xBuf == nil || outBuf == nil || xBuf.Ptr == 0 || outBuf.Ptr == 0 || xBuf.Size < batch*m.InDim*4 || outBuf.Size < batch*m.OutDim*4 || fnQ4KGemvBatch == 0 {
		return fmt.Errorf("invalid Q4_K batch buffer GEMV buffers")
	}
	inDim := uint32(m.InDim)
	outDim := uint32(m.OutDim)
	batchU := uint32(batch)
	args := []unsafe.Pointer{unsafe.Pointer(&xBuf.Ptr), unsafe.Pointer(&m.Q.Ptr), unsafe.Pointer(&m.Scales.Ptr), unsafe.Pointer(&m.Mins.Ptr), unsafe.Pointer(&outBuf.Ptr), unsafe.Pointer(&inDim), unsafe.Pointer(&outDim), unsafe.Pointer(&batchU)}
	return LaunchKernel(fnQ4KGemvBatch, uint32(m.OutDim), uint32(batch), 1, 256, 1, 1, 0, args...)
}

func gemmQ4RawUpstream(out, x *Buffer, batch int, m *GPUQ4KMatrix) error {
	q8, unlock, err := prepareQ4RawUpstream(x, batch, m.InDim)
	if err != nil {
		return err
	}
	defer unlock()
	return gemmQ4RawUpstreamPrepared(out, q8, batch, m)
}

func gemmQ4RawUpstreamPair(outA, outB, x *Buffer, batch int, a, b *GPUQ4KMatrix) error {
	if a == nil || b == nil || !a.raw || !b.raw || a.InDim != b.InDim {
		return fmt.Errorf("invalid raw Q4_K projection pair")
	}
	q8, unlock, err := prepareQ4RawUpstream(x, batch, a.InDim)
	if err != nil {
		return err
	}
	defer unlock()
	if err = gemmQ4RawUpstreamPrepared(outA, q8, batch, a); err != nil {
		return err
	}
	return gemmQ4RawUpstreamPrepared(outB, q8, batch, b)
}

func prepareQ4RawUpstream(x *Buffer, batch, inDim int) (*Buffer, func(), error) {
	blocks := inDim / 128
	tileRows := upstreamQ4TileRows(batch)
	stride := (batch + tileRows - 1) / tileRows * tileRows
	q8, unlock, err := q4UpstreamQ8(stride * blocks * 144)
	if err != nil {
		return nil, nil, err
	}
	kk, bb, ss := uint32(inDim), uint32(batch), uint32(stride)
	if err = LaunchKernel(fnQuantizeQ81MMQ, uint32(inDim/128), uint32(batch), 1, 32, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&q8.Ptr), unsafe.Pointer(&kk), unsafe.Pointer(&bb), unsafe.Pointer(&ss)); err != nil {
		unlock()
		return nil, nil, err
	}
	return q8, unlock, nil
}

func gemmQ4RawUpstreamPrepared(out, q8 *Buffer, batch int, m *GPUQ4KMatrix) error {
	var fn CUfunction
	switch {
	case batch <= 8:
		fn = fnQ4UpstreamMMQJ8
	case batch <= 16:
		fn = fnQ4UpstreamMMQJ16
	case batch <= 24:
		fn = fnQ4UpstreamMMQJ24
	case batch <= 32:
		fn = fnQ4UpstreamMMQJ32
	default:
		fn = fnQ4UpstreamMMQJ64
	}
	kk, nn, bb := uint32(m.InDim), uint32(m.OutDim), uint32(batch)
	return LaunchKernel(fn, uint32((m.OutDim+127)/128), uint32((batch+upstreamQ4TileRows(batch)-1)/upstreamQ4TileRows(batch)), 1, 32, 8, 1, uint32(upstreamQ4SharedBytes(batch)), unsafe.Pointer(&m.Q.Ptr), unsafe.Pointer(&q8.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&nn), unsafe.Pointer(&kk), unsafe.Pointer(&bb))
}

func GemmQ4CoalescedF32ToBuffer(outBuf, xBuf *Buffer, batch int, m *GPUQ4KMatrix) error {
	if batch <= 0 || m == nil || !m.coalesced || m.Q == nil || m.Scales == nil || m.Mins == nil || xBuf == nil || outBuf == nil || xBuf.Size < batch*m.InDim*4 || outBuf.Size < batch*m.OutDim*4 {
		return fmt.Errorf("invalid coalesced Q4_K F32 buffers")
	}
	kk, nn, bb := uint32(m.InDim), uint32(m.OutDim), uint32(batch)
	return LaunchKernel(fnQ4CoalescedF32, uint32((m.OutDim+3)/4), bb, 1, 128, 1, 1, 0, unsafe.Pointer(&xBuf.Ptr), unsafe.Pointer(&m.Q.Ptr), unsafe.Pointer(&m.Scales.Ptr), unsafe.Pointer(&m.Mins.Ptr), unsafe.Pointer(&outBuf.Ptr), unsafe.Pointer(&kk), unsafe.Pointer(&nn), unsafe.Pointer(&bb))
}

func GemmQ4CoalescedQ8ToBuffer(outBuf, xBuf *Buffer, batch int, m *GPUQ4KMatrix) error {
	if batch <= 0 || m == nil || !m.coalesced || m.Q == nil || m.Scales == nil || m.Mins == nil || xBuf == nil || outBuf == nil || xBuf.Size < batch*m.InDim*4 || outBuf.Size < batch*m.OutDim*4 {
		return fmt.Errorf("invalid coalesced Q4_K Q8 buffers")
	}
	groups := m.InDim / 32
	q, d, s, unlock, err := q8ProjectionBuffers(batch*m.InDim, batch*groups, batch*groups*4)
	if err != nil {
		return err
	}
	defer unlock()
	if err := quantizeQ8RowsSum(q, d, s, xBuf, batch, m.InDim); err != nil {
		return err
	}
	return gemmQ4CoalescedPrepared(outBuf, q, d, s, batch, m)
}

func gemmQ4PairPrepared(outA, outB, q, d, s *Buffer, batch int, a, b *GPUQ4KMatrix) error {
	if fnQ4PairMMQJ16 == 0 || a == nil || b == nil || a.InDim != b.InDim || a.OutDim != b.OutDim {
		return fmt.Errorf("invalid prepared Q4 pair")
	}
	kk, nn, bb := uint32(a.InDim), uint32(a.OutDim), uint32(batch)
	return LaunchKernel(fnQ4PairMMQJ16, uint32((a.OutDim+31)/32), uint32((batch+15)/16), 1, 256, 1, 1, 0, unsafe.Pointer(&q.Ptr), unsafe.Pointer(&d.Ptr), unsafe.Pointer(&s.Ptr), unsafe.Pointer(&a.Q.Ptr), unsafe.Pointer(&a.Scales.Ptr), unsafe.Pointer(&a.Mins.Ptr), unsafe.Pointer(&b.Q.Ptr), unsafe.Pointer(&b.Scales.Ptr), unsafe.Pointer(&b.Mins.Ptr), unsafe.Pointer(&outA.Ptr), unsafe.Pointer(&outB.Ptr), unsafe.Pointer(&kk), unsafe.Pointer(&nn), unsafe.Pointer(&bb))
}

func quantizeQ8RowsSum(q, d, s, x *Buffer, batch, inDim int) error {
	groups := inDim / 32
	rr, cc := uint32(batch), uint32(inDim)
	return LaunchKernel(fnQuantizeQ8RowsSum, uint32(groups), rr, 1, 32, 1, 1, 32*8, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&q.Ptr), unsafe.Pointer(&d.Ptr), unsafe.Pointer(&s.Ptr), unsafe.Pointer(&rr), unsafe.Pointer(&cc))
}
func gemmQ4CoalescedPrepared(outBuf, q, d, s *Buffer, batch int, m *GPUQ4KMatrix) error {
	kk, nn, bb := uint32(m.InDim), uint32(m.OutDim), uint32(batch)
	if batch >= 16 && fnQ4CoalescedMMQJ16 != 0 {
		return LaunchKernel(fnQ4CoalescedMMQJ16, uint32((m.OutDim+31)/32), uint32((batch+15)/16), 1, 256, 1, 1, 0, unsafe.Pointer(&q.Ptr), unsafe.Pointer(&d.Ptr), unsafe.Pointer(&s.Ptr), unsafe.Pointer(&m.Q.Ptr), unsafe.Pointer(&m.Scales.Ptr), unsafe.Pointer(&m.Mins.Ptr), unsafe.Pointer(&outBuf.Ptr), unsafe.Pointer(&kk), unsafe.Pointer(&nn), unsafe.Pointer(&bb))
	}
	if batch >= 4 && fnQ4CoalescedMMQ8 != 0 {
		return LaunchKernel(fnQ4CoalescedMMQ8, uint32((m.OutDim+31)/32), uint32((batch+7)/8), 1, 256, 1, 1, 0, unsafe.Pointer(&q.Ptr), unsafe.Pointer(&d.Ptr), unsafe.Pointer(&s.Ptr), unsafe.Pointer(&m.Q.Ptr), unsafe.Pointer(&m.Scales.Ptr), unsafe.Pointer(&m.Mins.Ptr), unsafe.Pointer(&outBuf.Ptr), unsafe.Pointer(&kk), unsafe.Pointer(&nn), unsafe.Pointer(&bb))
	}
	fn, tile := fnQ4CoalescedQ8, 16
	if fnQ4CoalescedQ8Batch8 != 0 {
		fn, tile = fnQ4CoalescedQ8Batch8, 8
	}
	return LaunchKernel(fn, uint32((m.OutDim+3)/4), uint32((batch+tile-1)/tile), 1, 128, 1, 1, 0, unsafe.Pointer(&q.Ptr), unsafe.Pointer(&d.Ptr), unsafe.Pointer(&s.Ptr), unsafe.Pointer(&m.Q.Ptr), unsafe.Pointer(&m.Scales.Ptr), unsafe.Pointer(&m.Mins.Ptr), unsafe.Pointer(&outBuf.Ptr), unsafe.Pointer(&kk), unsafe.Pointer(&nn), unsafe.Pointer(&bb))
}

func GemmQ4KQ8ToBuffer(outBuf, xBuf *Buffer, batch int, m *GPUQ4KMatrix) error {
	if batch <= 0 || m == nil || m.Q == nil || m.Scales == nil || m.Mins == nil || xBuf == nil || outBuf == nil || xBuf.Ptr == 0 || outBuf.Ptr == 0 || xBuf.Size < batch*m.InDim*4 || outBuf.Size < batch*m.OutDim*4 {
		return fmt.Errorf("invalid Q4_K Q8 GEMM buffers")
	}
	groups := (m.InDim + 31) / 32
	q, d, s, unlock, err := q8ProjectionBuffers(batch*m.InDim, batch*groups, batch*groups*4)
	if err != nil {
		return err
	}
	defer unlock()
	rr, cc := uint32(batch), uint32(m.InDim)
	if err := LaunchKernel(fnQuantizeQ8RowsSum, uint32(groups), rr, 1, 32, 1, 1, 32*8, unsafe.Pointer(&xBuf.Ptr), unsafe.Pointer(&q.Ptr), unsafe.Pointer(&d.Ptr), unsafe.Pointer(&s.Ptr), unsafe.Pointer(&rr), unsafe.Pointer(&cc)); err != nil {
		return err
	}
	kk, nn, bb := uint32(m.InDim), uint32(m.OutDim), uint32(batch)
	fn, tile := fnQ4KQ8Batch4, 4
	if batch >= 8 && fnQ4KQ8Batch8 != 0 {
		fn, tile = fnQ4KQ8Batch8, 8
	}
	if batch >= 16 && fnQ4KQ8Batch16 != 0 {
		fn, tile = fnQ4KQ8Batch16, 16
	}
	return LaunchKernel(fn, uint32((m.OutDim+3)/4), uint32((batch+tile-1)/tile), 1, 128, 1, 1, 0, unsafe.Pointer(&q.Ptr), unsafe.Pointer(&d.Ptr), unsafe.Pointer(&s.Ptr), unsafe.Pointer(&m.Q.Ptr), unsafe.Pointer(&m.Scales.Ptr), unsafe.Pointer(&m.Mins.Ptr), unsafe.Pointer(&outBuf.Ptr), unsafe.Pointer(&kk), unsafe.Pointer(&nn), unsafe.Pointer(&bb))
}

func GemmQ4KBatch8ToBuffer(outBuf, xBuf *Buffer, batch int, m *GPUQ4KMatrix) error {
	if batch <= 0 || m == nil || m.Q == nil || m.Scales == nil || m.Mins == nil || xBuf == nil || outBuf == nil || xBuf.Ptr == 0 || outBuf.Ptr == 0 || xBuf.Size < batch*m.InDim*4 || outBuf.Size < batch*m.OutDim*4 || fnQ4KGemmBatch8 == 0 {
		return fmt.Errorf("invalid Q4_K batch8 GEMM buffers")
	}
	kk, nn, bb := uint32(m.InDim), uint32(m.OutDim), uint32(batch)
	tiles := uint32((batch + 7) / 8)
	gridX := uint32((m.OutDim + 3) / 4)
	return LaunchKernel(fnQ4KGemmBatch8, gridX, tiles, 1, 128, 1, 1, 0, unsafe.Pointer(&xBuf.Ptr), unsafe.Pointer(&m.Q.Ptr), unsafe.Pointer(&m.Scales.Ptr), unsafe.Pointer(&m.Mins.Ptr), unsafe.Pointer(&outBuf.Ptr), unsafe.Pointer(&kk), unsafe.Pointer(&nn), unsafe.Pointer(&bb))
}

func GemvQ4KBatch(out, x []float32, batch int, m *GPUQ4KMatrix) error {
	if batch <= 0 {
		return nil
	}
	if m == nil || m.Q == nil || m.Scales == nil || m.Mins == nil || len(x) < batch*m.InDim || len(out) < batch*m.OutDim {
		return fmt.Errorf("invalid Q4_K batch GEMV buffers")
	}
	if fnQ4KGemvBatch == 0 {
		for b := 0; b < batch; b++ {
			if err := GemvQ4K(out[b*m.OutDim:(b+1)*m.OutDim], x[b*m.InDim:(b+1)*m.InDim], m); err != nil {
				return err
			}
		}
		return nil
	}
	xBuf, err := Malloc(batch * m.InDim)
	if err != nil {
		return err
	}
	defer xBuf.Free()
	outBuf, err := Malloc(batch * m.OutDim)
	if err != nil {
		return err
	}
	defer outBuf.Free()
	if err := xBuf.Upload(x[:batch*m.InDim]); err != nil {
		return err
	}
	inDim := uint32(m.InDim)
	outDim := uint32(m.OutDim)
	batchU := uint32(batch)
	args := []unsafe.Pointer{unsafe.Pointer(&xBuf.Ptr), unsafe.Pointer(&m.Q.Ptr), unsafe.Pointer(&m.Scales.Ptr), unsafe.Pointer(&m.Mins.Ptr), unsafe.Pointer(&outBuf.Ptr), unsafe.Pointer(&inDim), unsafe.Pointer(&outDim), unsafe.Pointer(&batchU)}
	if err := LaunchKernel(fnQ4KGemvBatch, uint32(m.OutDim), uint32(batch), 1, 256, 1, 1, 0, args...); err != nil {
		return err
	}
	return outBuf.Download(out[:batch*m.OutDim])
}

func GemvQ4K(out, x []float32, m *GPUQ4KMatrix) error {
	if m == nil || m.Q == nil || m.Scales == nil || m.Mins == nil || len(x) < m.InDim || len(out) < m.OutDim || fnQ4KGemv == 0 {
		return fmt.Errorf("invalid Q4_K GEMV buffers")
	}
	xBuf, err := Malloc(m.InDim)
	if err != nil {
		return err
	}
	defer xBuf.Free()
	outBuf, err := Malloc(m.OutDim)
	if err != nil {
		return err
	}
	defer outBuf.Free()
	if err := xBuf.Upload(x[:m.InDim]); err != nil {
		return err
	}
	inDim := uint32(m.InDim)
	outDim := uint32(m.OutDim)
	args := []unsafe.Pointer{unsafe.Pointer(&xBuf.Ptr), unsafe.Pointer(&m.Q.Ptr), unsafe.Pointer(&m.Scales.Ptr), unsafe.Pointer(&m.Mins.Ptr), unsafe.Pointer(&outBuf.Ptr), unsafe.Pointer(&inDim), unsafe.Pointer(&outDim)}
	if err := LaunchKernel(fnQ4KGemv, uint32(m.OutDim), 1, 1, 256, 1, 1, 0, args...); err != nil {
		return err
	}
	return outBuf.Download(out[:m.OutDim])
}
