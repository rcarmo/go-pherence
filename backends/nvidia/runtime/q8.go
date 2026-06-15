package nvidia

import (
	"encoding/binary"
	"fmt"
	"unsafe"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/internal/checked"
)

var (
	fnQ8_0Gemv                  CUfunction
	fnQ8_0GemvBatch             CUfunction
	fnQ8_0GemvScatter           CUfunction
	fnQ8_0GemvScatterByWork     CUfunction
	fnQ8_0GemvScatterByWorkPtrs CUfunction
)

type GPUQ8_0Matrix struct {
	Q      *Buffer // int8 bytes stored in device memory; Size is bytes
	Scales *Buffer // F32 scales [outDim, inDim/32]
	InDim  int
	OutDim int
}

type GPUQ8_0PointerTable struct {
	QPtrs     *Buffer // uint64 device pointers, one per active expert
	ScalePtrs *Buffer // uint64 device pointers, one per active expert
	InDim     int
	OutDim    int
	Count     int
}

func UploadQ8_0PointerTable(mats []*GPUQ8_0Matrix) (*GPUQ8_0PointerTable, error) {
	if len(mats) == 0 {
		return nil, fmt.Errorf("empty Q8_0 pointer table")
	}
	inDim, outDim := mats[0].InDim, mats[0].OutDim
	qPtrs := make([]byte, len(mats)*8)
	sPtrs := make([]byte, len(mats)*8)
	for i, m := range mats {
		if m == nil || m.Q == nil || m.Scales == nil || m.Q.Ptr == 0 || m.Scales.Ptr == 0 || m.InDim != inDim || m.OutDim != outDim {
			return nil, fmt.Errorf("invalid Q8_0 matrix %d for pointer table", i)
		}
		binary.LittleEndian.PutUint64(qPtrs[i*8:(i+1)*8], uint64(m.Q.Ptr))
		binary.LittleEndian.PutUint64(sPtrs[i*8:(i+1)*8], uint64(m.Scales.Ptr))
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
	return &GPUQ8_0PointerTable{QPtrs: qBuf, ScalePtrs: sBuf, InDim: inDim, OutDim: outDim, Count: len(mats)}, nil
}

func (t *GPUQ8_0PointerTable) Free() {
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
}

func unpackQ8_0MatrixRows(raw []byte, inDim, outDim int) ([]byte, []float32, error) {
	if inDim <= 0 || outDim <= 0 || inDim%32 != 0 {
		return nil, nil, fmt.Errorf("invalid Q8_0 dims in=%d out=%d", inDim, outDim)
	}
	blocks := inDim / 32
	rowBytes := blocks * 34
	needRaw, okRaw := checked.MulInt(rowBytes, outDim)
	qLen, okQ := checked.MulInt(inDim, outDim)
	sLen, okS := checked.MulInt(blocks, outDim)
	if !okRaw || !okQ || !okS || len(raw) < needRaw {
		return nil, nil, fmt.Errorf("invalid Q8_0 raw len=%d need=%d", len(raw), needRaw)
	}
	q := make([]byte, qLen)
	scales := make([]float32, sLen)
	for r := 0; r < outDim; r++ {
		row := raw[r*rowBytes : (r+1)*rowBytes]
		for b := 0; b < blocks; b++ {
			blk := row[b*34:]
			scales[r*blocks+b] = half.F16ToF32(binary.LittleEndian.Uint16(blk[:2]))
			copy(q[r*inDim+b*32:r*inDim+(b+1)*32], blk[2:34])
		}
	}
	return q, scales, nil
}

func UploadQ8_0MatrixRows(raw []byte, inDim, outDim int) (*GPUQ8_0Matrix, error) {
	q, scales, err := unpackQ8_0MatrixRows(raw, inDim, outDim)
	if err != nil {
		return nil, err
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
	return &GPUQ8_0Matrix{Q: qBuf, Scales: sBuf, InDim: inDim, OutDim: outDim}, nil
}

func UploadQ8_0MatrixRowsInto(m *GPUQ8_0Matrix, raw []byte, inDim, outDim int) error {
	if m == nil || m.Q == nil || m.Scales == nil || m.Q.Ptr == 0 || m.Scales.Ptr == 0 || m.InDim != inDim || m.OutDim != outDim {
		return fmt.Errorf("invalid destination Q8_0 matrix for in-place upload")
	}
	q, scales, err := unpackQ8_0MatrixRows(raw, inDim, outDim)
	if err != nil {
		return err
	}
	if m.Q.Size < len(q) || m.Scales.Size < len(scales)*4 {
		return fmt.Errorf("destination Q8_0 matrix too small q=%d/%d scales=%d/%d", m.Q.Size, len(q), m.Scales.Size, len(scales)*4)
	}
	if err := m.Q.UploadBytes(q); err != nil {
		return err
	}
	return m.Scales.Upload(scales)
}

func (m *GPUQ8_0Matrix) Free() {
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
}

func MallocBytes(n int) (*Buffer, error) {
	if n <= 0 {
		return &Buffer{}, nil
	}
	elems := (n + 3) / 4
	buf, err := Malloc(elems)
	if err != nil {
		return nil, err
	}
	buf.Size = n
	return buf, nil
}

func GemvQ8_0ScatterByWorkPtrs(dstBuf, xBuf, workActive, workPos, workWeight *Buffer, workLen int, table *GPUQ8_0PointerTable) error {
	if workLen <= 0 {
		return nil
	}
	if table == nil || table.QPtrs == nil || table.ScalePtrs == nil || table.Count <= 0 || table.InDim <= 0 || table.OutDim <= 0 || dstBuf == nil || xBuf == nil || workActive == nil || workPos == nil || workWeight == nil || dstBuf.Ptr == 0 || xBuf.Ptr == 0 || workActive.Ptr == 0 || workPos.Ptr == 0 || workWeight.Ptr == 0 || table.QPtrs.Ptr == 0 || table.ScalePtrs.Ptr == 0 || xBuf.Size < workLen*table.InDim*4 || workActive.Size < workLen*4 || workPos.Size < workLen*4 || workWeight.Size < workLen*4 || fnQ8_0GemvScatterByWorkPtrs == 0 {
		return fmt.Errorf("invalid Q8_0 scatter-by-work pointer-table buffers")
	}
	inDim := uint32(table.InDim)
	expertOut := uint32(table.OutDim)
	work := uint32(workLen)
	active := uint32(table.Count)
	args := []unsafe.Pointer{unsafe.Pointer(&xBuf.Ptr), unsafe.Pointer(&workActive.Ptr), unsafe.Pointer(&workPos.Ptr), unsafe.Pointer(&workWeight.Ptr), unsafe.Pointer(&table.QPtrs.Ptr), unsafe.Pointer(&table.ScalePtrs.Ptr), unsafe.Pointer(&dstBuf.Ptr), unsafe.Pointer(&inDim), unsafe.Pointer(&expertOut), unsafe.Pointer(&work), unsafe.Pointer(&active)}
	return LaunchKernel(fnQ8_0GemvScatterByWorkPtrs, uint32(table.OutDim), uint32(workLen), 1, 256, 1, 1, 0, args...)
}

func GemvQ8_0ScatterByWork(dstBuf, xBuf, workActive, workPos, workWeight *Buffer, workLen, activeExperts int, m *GPUQ8_0Matrix) error {
	if workLen <= 0 {
		return nil
	}
	if activeExperts <= 0 || m == nil || m.Q == nil || m.Scales == nil || dstBuf == nil || xBuf == nil || workActive == nil || workPos == nil || workWeight == nil || dstBuf.Ptr == 0 || xBuf.Ptr == 0 || workActive.Ptr == 0 || workPos.Ptr == 0 || workWeight.Ptr == 0 || xBuf.Size < workLen*m.InDim*4 || workActive.Size < workLen*4 || workPos.Size < workLen*4 || workWeight.Size < workLen*4 || m.OutDim <= 0 || m.OutDim%activeExperts != 0 || fnQ8_0GemvScatterByWork == 0 {
		return fmt.Errorf("invalid Q8_0 scatter-by-work buffers")
	}
	expertOutDim := m.OutDim / activeExperts
	inDim := uint32(m.InDim)
	matrixRows := uint32(m.OutDim)
	expertOut := uint32(expertOutDim)
	work := uint32(workLen)
	active := uint32(activeExperts)
	args := []unsafe.Pointer{unsafe.Pointer(&xBuf.Ptr), unsafe.Pointer(&workActive.Ptr), unsafe.Pointer(&workPos.Ptr), unsafe.Pointer(&workWeight.Ptr), unsafe.Pointer(&m.Q.Ptr), unsafe.Pointer(&m.Scales.Ptr), unsafe.Pointer(&dstBuf.Ptr), unsafe.Pointer(&inDim), unsafe.Pointer(&matrixRows), unsafe.Pointer(&expertOut), unsafe.Pointer(&work), unsafe.Pointer(&active)}
	return LaunchKernel(fnQ8_0GemvScatterByWork, uint32(expertOutDim), uint32(workLen), 1, 256, 1, 1, 0, args...)
}

func GemvQ8_0BatchScatter(dstBuf, xBuf, posBuf, weightBuf *Buffer, batch int, m *GPUQ8_0Matrix) error {
	if batch <= 0 {
		return nil
	}
	if m == nil || m.Q == nil || m.Scales == nil || dstBuf == nil || xBuf == nil || posBuf == nil || weightBuf == nil || dstBuf.Ptr == 0 || xBuf.Ptr == 0 || posBuf.Ptr == 0 || weightBuf.Ptr == 0 || xBuf.Size < batch*m.InDim*4 || posBuf.Size < batch*4 || weightBuf.Size < batch*4 || fnQ8_0GemvScatter == 0 {
		return fmt.Errorf("invalid Q8_0 batch scatter GEMV buffers")
	}
	inDim := uint32(m.InDim)
	outDim := uint32(m.OutDim)
	batchU := uint32(batch)
	args := []unsafe.Pointer{unsafe.Pointer(&xBuf.Ptr), unsafe.Pointer(&m.Q.Ptr), unsafe.Pointer(&m.Scales.Ptr), unsafe.Pointer(&dstBuf.Ptr), unsafe.Pointer(&posBuf.Ptr), unsafe.Pointer(&weightBuf.Ptr), unsafe.Pointer(&inDim), unsafe.Pointer(&outDim), unsafe.Pointer(&batchU)}
	return LaunchKernel(fnQ8_0GemvScatter, uint32(m.OutDim), uint32(batch), 1, 256, 1, 1, 0, args...)
}

func GemvQ8_0BatchToBuffer(outBuf *Buffer, xBuf *Buffer, batch int, m *GPUQ8_0Matrix) error {
	if batch <= 0 {
		return nil
	}
	if m == nil || m.Q == nil || m.Scales == nil || xBuf == nil || outBuf == nil || xBuf.Ptr == 0 || outBuf.Ptr == 0 || xBuf.Size < batch*m.InDim*4 || outBuf.Size < batch*m.OutDim*4 || fnQ8_0GemvBatch == 0 {
		return fmt.Errorf("invalid Q8_0 batch buffer GEMV buffers")
	}
	inDim := uint32(m.InDim)
	outDim := uint32(m.OutDim)
	batchU := uint32(batch)
	args := []unsafe.Pointer{unsafe.Pointer(&xBuf.Ptr), unsafe.Pointer(&m.Q.Ptr), unsafe.Pointer(&m.Scales.Ptr), unsafe.Pointer(&outBuf.Ptr), unsafe.Pointer(&inDim), unsafe.Pointer(&outDim), unsafe.Pointer(&batchU)}
	return LaunchKernel(fnQ8_0GemvBatch, uint32(m.OutDim), uint32(batch), 1, 256, 1, 1, 0, args...)
}

func GemvQ8_0Batch(out, x []float32, batch int, m *GPUQ8_0Matrix) error {
	if batch <= 0 {
		return nil
	}
	if m == nil || m.Q == nil || m.Scales == nil || len(x) < batch*m.InDim || len(out) < batch*m.OutDim {
		return fmt.Errorf("invalid Q8_0 batch GEMV buffers")
	}
	if fnQ8_0GemvBatch == 0 {
		for b := 0; b < batch; b++ {
			if err := GemvQ8_0(out[b*m.OutDim:(b+1)*m.OutDim], x[b*m.InDim:(b+1)*m.InDim], m); err != nil {
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
	args := []unsafe.Pointer{unsafe.Pointer(&xBuf.Ptr), unsafe.Pointer(&m.Q.Ptr), unsafe.Pointer(&m.Scales.Ptr), unsafe.Pointer(&outBuf.Ptr), unsafe.Pointer(&inDim), unsafe.Pointer(&outDim), unsafe.Pointer(&batchU)}
	if err := LaunchKernel(fnQ8_0GemvBatch, uint32(m.OutDim), uint32(batch), 1, 256, 1, 1, 0, args...); err != nil {
		return err
	}
	return outBuf.Download(out[:batch*m.OutDim])
}

func GemvQ8_0(out, x []float32, m *GPUQ8_0Matrix) error {
	if m == nil || m.Q == nil || m.Scales == nil || len(x) < m.InDim || len(out) < m.OutDim || fnQ8_0Gemv == 0 {
		return fmt.Errorf("invalid Q8_0 GEMV buffers")
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
	args := []unsafe.Pointer{unsafe.Pointer(&xBuf.Ptr), unsafe.Pointer(&m.Q.Ptr), unsafe.Pointer(&m.Scales.Ptr), unsafe.Pointer(&outBuf.Ptr), unsafe.Pointer(&inDim), unsafe.Pointer(&outDim)}
	if err := LaunchKernel(fnQ8_0Gemv, uint32(m.OutDim), 1, 1, 256, 1, 1, 0, args...); err != nil {
		return err
	}
	return outBuf.Download(out[:m.OutDim])
}
