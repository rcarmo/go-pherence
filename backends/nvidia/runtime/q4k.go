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
	fnQ4KGateUpGELU           CUfunction
	fnQ4KGateUpGELUByWork     CUfunction
	fnQ4KGateUpGELUByWorkPtrs CUfunction
)

type GPUQ4KMatrix struct {
	Q      *Buffer // packed q bytes [outDim, inDim/256, 128]
	Scales *Buffer // [outDim, inDim/256, 8]
	Mins   *Buffer // [outDim, inDim/256, 8]
	InDim  int
	OutDim int
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
	q, scales, mins, err := unpackQ4KMatrixRows(raw, inDim, outDim)
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
	return &GPUQ4KMatrix{Q: qBuf, Scales: sBuf, Mins: mBuf, InDim: inDim, OutDim: outDim}, nil
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
