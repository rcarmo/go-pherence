package nvidia

import (
	"encoding/binary"
	"fmt"
	"unsafe"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/internal/checked"
)

var fnQ5_0GemvBatch CUfunction

type GPUQ5_0Matrix struct {
	Q      *Buffer // packed low nibbles [outDim, inDim/32, 16]
	High   *Buffer // uint32 high-bit masks [outDim, inDim/32]
	Scales *Buffer // F32 scales [outDim, inDim/32]
	InDim  int
	OutDim int
}

func unpackQ5_0MatrixRows(raw []byte, inDim, outDim int) ([]byte, []uint32, []float32, error) {
	if inDim <= 0 || outDim <= 0 || inDim%32 != 0 {
		return nil, nil, nil, fmt.Errorf("invalid Q5_0 dims in=%d out=%d", inDim, outDim)
	}
	blocks := inDim / 32
	rowBytes := blocks * 22
	needRaw, okRaw := checked.MulInt(rowBytes, outDim)
	qLen, okQ := checked.MulInt(outDim*blocks, 16)
	hLen, okH := checked.MulInt(outDim, blocks)
	if !okRaw || !okQ || !okH || len(raw) < needRaw {
		return nil, nil, nil, fmt.Errorf("invalid Q5_0 raw len=%d need=%d", len(raw), needRaw)
	}
	q := make([]byte, qLen)
	high := make([]uint32, hLen)
	scales := make([]float32, hLen)
	for r := 0; r < outDim; r++ {
		row := raw[r*rowBytes : (r+1)*rowBytes]
		for b := 0; b < blocks; b++ {
			blk := row[b*22:]
			i := r*blocks + b
			scales[i] = half.F16ToF32(binary.LittleEndian.Uint16(blk[:2]))
			high[i] = binary.LittleEndian.Uint32(blk[2:6])
			copy(q[i*16:(i+1)*16], blk[6:22])
		}
	}
	return q, high, scales, nil
}

func UploadQ5_0MatrixRows(raw []byte, inDim, outDim int) (*GPUQ5_0Matrix, error) {
	q, high, scales, err := unpackQ5_0MatrixRows(raw, inDim, outDim)
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
	hBuf, err := MallocBytes(len(high) * 4)
	if err != nil {
		qBuf.Free()
		return nil, err
	}
	if err := hBuf.UploadUint32(high); err != nil {
		qBuf.Free()
		hBuf.Free()
		return nil, err
	}
	sBuf, err := Malloc(len(scales))
	if err != nil {
		qBuf.Free()
		hBuf.Free()
		return nil, err
	}
	if err := sBuf.Upload(scales); err != nil {
		qBuf.Free()
		hBuf.Free()
		sBuf.Free()
		return nil, err
	}
	return &GPUQ5_0Matrix{Q: qBuf, High: hBuf, Scales: sBuf, InDim: inDim, OutDim: outDim}, nil
}

func (m *GPUQ5_0Matrix) Free() {
	if m == nil {
		return
	}
	if m.Q != nil {
		m.Q.Free()
		m.Q = nil
	}
	if m.High != nil {
		m.High.Free()
		m.High = nil
	}
	if m.Scales != nil {
		m.Scales.Free()
		m.Scales = nil
	}
}

func GemvQ5_0Batch(out, x []float32, batch int, m *GPUQ5_0Matrix) error {
	if batch <= 0 {
		return nil
	}
	if m == nil || m.Q == nil || m.High == nil || m.Scales == nil || len(x) < batch*m.InDim || len(out) < batch*m.OutDim || fnQ5_0GemvBatch == 0 {
		return fmt.Errorf("invalid Q5_0 batch GEMV buffers")
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
	args := []unsafe.Pointer{unsafe.Pointer(&xBuf.Ptr), unsafe.Pointer(&m.Q.Ptr), unsafe.Pointer(&m.High.Ptr), unsafe.Pointer(&m.Scales.Ptr), unsafe.Pointer(&outBuf.Ptr), unsafe.Pointer(&inDim), unsafe.Pointer(&outDim), unsafe.Pointer(&batchU)}
	if err := LaunchKernel(fnQ5_0GemvBatch, uint32(m.OutDim), uint32(batch), 1, 256, 1, 1, 0, args...); err != nil {
		return err
	}
	return outBuf.Download(out[:batch*m.OutDim])
}
