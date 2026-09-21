package nvidia

import (
	"encoding/binary"
	"fmt"
	"unsafe"

	"github.com/rcarmo/go-pherence/half"
)

var (
	fnQ5KGemvBatch     CUfunction
	fnQ5KGemmWarp      CUfunction
	fnQ5PackedF32      CUfunction
	fnQ5PackedQ8       CUfunction
	fnQ6KGemvBatch     CUfunction
	fnQ6KGemmWarp      CUfunction
	fnQ6KGemmBatch8    CUfunction
	fnQuantizeQ8Rows   CUfunction
	fnQ6KQ8Batch4      CUfunction
	fnQuantizeQ8Rows16 CUfunction
	fnQ6PackedQ8       CUfunction
	fnQ6PackedF32      CUfunction
)

// GPUQKMatrix owns canonical row-major GGUF Q5_K or Q6_K bytes on the device.
type GPUQKMatrix struct {
	Raw         *Buffer
	PackedQ     *Buffer
	PackedScale *Buffer
	PackedMin   *Buffer
	InDim       int
	OutDim      int
	BlockSize   int
	kind        uint8
}

const (
	qkKindQ5K uint8 = 5
	qkKindQ6K uint8 = 6
)

func UploadQ5KMatrixRows(raw []byte, inDim, outDim int) (*GPUQKMatrix, error) {
	return uploadQKMatrixRows(raw, inDim, outDim, 176, qkKindQ5K)
}

func UploadQ6KMatrixRows(raw []byte, inDim, outDim int) (*GPUQKMatrix, error) {
	return uploadQKMatrixRows(raw, inDim, outDim, 210, qkKindQ6K)
}

func uploadQKMatrixRows(raw []byte, inDim, outDim, blockSize int, kind uint8) (*GPUQKMatrix, error) {
	if inDim <= 0 || outDim <= 0 || inDim%256 != 0 {
		return nil, fmt.Errorf("invalid Q%d_K dims in=%d out=%d", kind, inDim, outDim)
	}
	need, ok := checkedQKBytes(inDim, outDim, blockSize)
	if !ok || len(raw) != need {
		return nil, fmt.Errorf("invalid Q%d_K raw len=%d want=%d", kind, len(raw), need)
	}
	buf, err := MallocBytes(need)
	if err != nil {
		return nil, err
	}
	if err := buf.UploadBytes(raw); err != nil {
		buf.Free()
		return nil, err
	}
	m := &GPUQKMatrix{Raw: buf, InDim: inDim, OutDim: outDim, BlockSize: blockSize, kind: kind}
	if kind == qkKindQ5K {
		if err := m.packQ5(raw); err != nil {
			m.Free()
			return nil, err
		}
		m.Raw.Free()
		m.Raw = nil
	}
	if kind == qkKindQ6K {
		if err := m.packQ6(raw); err != nil {
			m.Free()
			return nil, err
		}
		m.Raw.Free()
		m.Raw = nil
	}
	return m, nil
}

func checkedQKBytes(inDim, outDim, blockSize int) (int, bool) {
	if inDim <= 0 || outDim <= 0 || blockSize <= 0 || inDim%256 != 0 {
		return 0, false
	}
	blocks := inDim / 256
	maxInt := int(^uint(0) >> 1)
	if blocks > maxInt/blockSize {
		return 0, false
	}
	row := blocks * blockSize
	if outDim > maxInt/row {
		return 0, false
	}
	return outDim * row, true
}

func (m *GPUQKMatrix) Free() {
	if m == nil {
		return
	}
	for _, b := range []*Buffer{m.Raw, m.PackedQ, m.PackedScale, m.PackedMin} {
		if b != nil {
			b.Free()
		}
	}
	m.Raw = nil
	m.PackedQ = nil
	m.PackedScale = nil
	m.PackedMin = nil
}

func (m *GPUQKMatrix) packQ5(raw []byte) error {
	groups := m.InDim / 32
	q := make([]byte, m.OutDim*m.InDim)
	scales := make([]float32, m.OutDim*groups)
	mins := make([]float32, len(scales))
	blocks := m.InDim / 256
	for r := 0; r < m.OutDim; r++ {
		for b := 0; b < blocks; b++ {
			blk := raw[(r*blocks+b)*176:]
			d := half.F16ToF32(binary.LittleEndian.Uint16(blk[:2]))
			dm := half.F16ToF32(binary.LittleEndian.Uint16(blk[2:4]))
			s, qh, ql := blk[4:16], blk[16:48], blk[48:176]
			var ss, mm [8]float32
			for j := 0; j < 4; j++ {
				ss[j] = float32(s[j]&63) * d
				mm[j] = float32(s[j+4]&63) * dm
			}
			for j := 4; j < 8; j++ {
				k := j - 4
				ss[j] = float32((s[j+4]&15)|((s[k]>>6)<<4)) * d
				mm[j] = float32((s[j+4]>>4)|((s[k+4]>>6)<<4)) * dm
			}
			for g := 0; g < 4; g++ {
				for i := 0; i < 32; i++ {
					q0, q1 := ql[g*32+i]&15, ql[g*32+i]>>4
					if qh[i]&(1<<uint(g*2)) != 0 {
						q0 += 16
					}
					if qh[i]&(1<<uint(g*2+1)) != 0 {
						q1 += 16
					}
					base := r*m.InDim + b*256 + g*64
					q[base+i] = q0
					q[base+32+i] = q1
				}
				scales[r*groups+b*8+g*2] = ss[g*2]
				mins[r*groups+b*8+g*2] = mm[g*2]
				scales[r*groups+b*8+g*2+1] = ss[g*2+1]
				mins[r*groups+b*8+g*2+1] = mm[g*2+1]
			}
		}
	}
	var err error
	m.PackedQ, err = MallocBytes(len(q))
	if err != nil {
		return err
	}
	if err = m.PackedQ.UploadBytes(q); err != nil {
		return err
	}
	m.PackedScale, err = Malloc(len(scales))
	if err != nil {
		return err
	}
	if err = m.PackedScale.Upload(scales); err != nil {
		return err
	}
	m.PackedMin, err = Malloc(len(mins))
	if err != nil {
		return err
	}
	return m.PackedMin.Upload(mins)
}

func (m *GPUQKMatrix) packQ6(raw []byte) error {
	groups := m.InDim / 16
	q := make([]byte, m.OutDim*m.InDim)
	sc := make([]float32, m.OutDim*groups)
	blocks := m.InDim / 256
	for r := 0; r < m.OutDim; r++ {
		for b := 0; b < blocks; b++ {
			blk := raw[(r*blocks+b)*210:]
			d := half.F16ToF32(binary.LittleEndian.Uint16(blk[208:210]))
			for i := 0; i < 256; i++ {
				halfBlock, group, lane := i/128, (i%128)/32, i%32
				v := blk[halfBlock*64+(group%2)*32+lane]
				lo := v & 15
				if group >= 2 {
					lo = v >> 4
				}
				hi := (blk[128+halfBlock*32+lane] >> uint(group*2)) & 3
				qq := int8(lo|(hi<<4)) - 32
				q[r*m.InDim+b*256+i] = byte(qq)
				scale := int8(blk[192+halfBlock*8+(lane/16)+group*2])
				sc[r*groups+b*16+i/16] = d * float32(scale)
			}
		}
	}
	var err error
	m.PackedQ, err = MallocBytes(len(q))
	if err != nil {
		return err
	}
	if err = m.PackedQ.UploadBytes(q); err != nil {
		return err
	}
	m.PackedScale, err = Malloc(len(sc))
	if err != nil {
		return err
	}
	return m.PackedScale.Upload(sc)
}

func GemvQ5KBatchToBuffer(out, x *Buffer, batch int, m *GPUQKMatrix) error {
	if batch >= 4 && m != nil && m.PackedQ != nil && m.PackedScale != nil && m.PackedMin != nil && fnQuantizeQ8RowsSum != 0 && fnQ5PackedQ8 != 0 {
		return gemmQ5PackedQ8ToBuffer(out, x, batch, m)
	}
	if m != nil && m.PackedQ != nil && m.PackedScale != nil && m.PackedMin != nil && fnQ5PackedF32 != 0 {
		return gemmQ5PackedF32ToBuffer(out, x, batch, m)
	}
	if fnQ5KGemmWarp != 0 {
		return gemmQ5KWarpToBuffer(out, x, batch, m)
	}
	return gemvQKBatchToBuffer(out, x, batch, m, qkKindQ5K, fnQ5KGemvBatch)
}

func gemmQ5PackedF32ToBuffer(out, x *Buffer, batch int, m *GPUQKMatrix) error {
	kk, nn, bb := uint32(m.InDim), uint32(m.OutDim), uint32(batch)
	return LaunchKernel(fnQ5PackedF32, uint32((m.OutDim+3)/4), bb, 1, 128, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&m.PackedQ.Ptr), unsafe.Pointer(&m.PackedScale.Ptr), unsafe.Pointer(&m.PackedMin.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&kk), unsafe.Pointer(&nn), unsafe.Pointer(&bb))
}
func gemmQ5PackedQ8ToBuffer(out, x *Buffer, batch int, m *GPUQKMatrix) error {
	groups := m.InDim / 32
	q, d, s, unlock, err := q8ProjectionBuffers(batch*m.InDim, batch*groups, batch*groups*4)
	if err != nil {
		return err
	}
	defer unlock()
	rr, cc := uint32(batch), uint32(m.InDim)
	if err := LaunchKernel(fnQuantizeQ8RowsSum, uint32(groups), rr, 1, 32, 1, 1, 32*8, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&q.Ptr), unsafe.Pointer(&d.Ptr), unsafe.Pointer(&s.Ptr), unsafe.Pointer(&rr), unsafe.Pointer(&cc)); err != nil {
		return err
	}
	kk, nn, bb := uint32(m.InDim), uint32(m.OutDim), uint32(batch)
	return LaunchKernel(fnQ5PackedQ8, uint32((m.OutDim+3)/4), uint32((batch+3)/4), 1, 128, 1, 1, 0, unsafe.Pointer(&q.Ptr), unsafe.Pointer(&d.Ptr), unsafe.Pointer(&s.Ptr), unsafe.Pointer(&m.PackedQ.Ptr), unsafe.Pointer(&m.PackedScale.Ptr), unsafe.Pointer(&m.PackedMin.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&kk), unsafe.Pointer(&nn), unsafe.Pointer(&bb))
}

func gemmQ5KWarpToBuffer(out, x *Buffer, batch int, m *GPUQKMatrix) error {
	if batch <= 0 || m == nil || m.kind != qkKindQ5K || m.Raw == nil || x == nil || out == nil || m.Raw.Ptr == 0 || x.Ptr == 0 || out.Ptr == 0 || x.Size < batch*m.InDim*4 || out.Size < batch*m.OutDim*4 {
		return fmt.Errorf("invalid Q5_K warp GEMM buffers")
	}
	kk, nn, bb := uint32(m.InDim), uint32(m.OutDim), uint32(batch)
	return LaunchKernel(fnQ5KGemmWarp, uint32((m.OutDim+3)/4), bb, 1, 128, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&m.Raw.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&kk), unsafe.Pointer(&nn), unsafe.Pointer(&bb))
}

func GemvQ6KBatchToBuffer(out, x *Buffer, batch int, m *GPUQKMatrix) error {
	if batch >= 4 && m != nil && m.PackedQ != nil && m.PackedScale != nil && fnQuantizeQ8Rows16 != 0 && fnQ6PackedQ8 != 0 {
		return gemmQ6PackedQ8ToBuffer(out, x, batch, m)
	}
	if m != nil && m.PackedQ != nil && m.PackedScale != nil && fnQ6PackedF32 != 0 {
		return gemmQ6PackedF32ToBuffer(out, x, batch, m)
	}
	if batch >= 4 && fnQuantizeQ8Rows != 0 && fnQ6KQ8Batch4 != 0 && m.Raw != nil {
		return gemmQ6KQ8ToBuffer(out, x, batch, m)
	}
	if batch >= 4 && fnQ6KGemmBatch8 != 0 {
		return gemmQ6KBatch8ToBuffer(out, x, batch, m)
	}
	if fnQ6KGemmWarp != 0 {
		return gemmQ6KWarpToBuffer(out, x, batch, m)
	}
	return gemvQKBatchToBuffer(out, x, batch, m, qkKindQ6K, fnQ6KGemvBatch)
}

func gemmQ6PackedF32ToBuffer(out, x *Buffer, batch int, m *GPUQKMatrix) error {
	kk, nn, bb := uint32(m.InDim), uint32(m.OutDim), uint32(batch)
	return LaunchKernel(fnQ6PackedF32, uint32((m.OutDim+3)/4), bb, 1, 128, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&m.PackedQ.Ptr), unsafe.Pointer(&m.PackedScale.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&kk), unsafe.Pointer(&nn), unsafe.Pointer(&bb))
}

func gemmQ6PackedQ8ToBuffer(out, x *Buffer, batch int, m *GPUQKMatrix) error {
	groups := m.InDim / 16
	q, d, _, unlock, err := q8ProjectionBuffers(batch*m.InDim, batch*groups, 0)
	if err != nil {
		return err
	}
	defer unlock()
	rr, cc := uint32(batch), uint32(m.InDim)
	if err := LaunchKernel(fnQuantizeQ8Rows16, uint32(groups), rr, 1, 16, 1, 1, 16*4, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&q.Ptr), unsafe.Pointer(&d.Ptr), unsafe.Pointer(&rr), unsafe.Pointer(&cc)); err != nil {
		return err
	}
	kk, nn, bb := uint32(m.InDim), uint32(m.OutDim), uint32(batch)
	return LaunchKernel(fnQ6PackedQ8, uint32((m.OutDim+3)/4), uint32((batch+3)/4), 1, 128, 1, 1, 0, unsafe.Pointer(&q.Ptr), unsafe.Pointer(&d.Ptr), unsafe.Pointer(&m.PackedQ.Ptr), unsafe.Pointer(&m.PackedScale.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&kk), unsafe.Pointer(&nn), unsafe.Pointer(&bb))
}

func gemmQ6KQ8ToBuffer(out, x *Buffer, batch int, m *GPUQKMatrix) error {
	if batch <= 0 || m == nil || m.kind != qkKindQ6K || m.Raw == nil || x == nil || out == nil || m.Raw.Ptr == 0 || x.Ptr == 0 || out.Ptr == 0 || x.Size < batch*m.InDim*4 || out.Size < batch*m.OutDim*4 {
		return fmt.Errorf("invalid Q6_K Q8 GEMM buffers")
	}
	groups := (m.InDim + 31) / 32
	q, d, _, unlock, err := q8ProjectionBuffers(batch*m.InDim, batch*groups, 0)
	if err != nil {
		return err
	}
	defer unlock()
	rr, cc := uint32(batch), uint32(m.InDim)
	if err := LaunchKernel(fnQuantizeQ8Rows, uint32(groups), rr, 1, 32, 1, 1, 32*4, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&q.Ptr), unsafe.Pointer(&d.Ptr), unsafe.Pointer(&rr), unsafe.Pointer(&cc)); err != nil {
		return err
	}
	kk, nn, bb := uint32(m.InDim), uint32(m.OutDim), uint32(batch)
	return LaunchKernel(fnQ6KQ8Batch4, uint32((m.OutDim+3)/4), uint32((batch+3)/4), 1, 128, 1, 1, 0, unsafe.Pointer(&q.Ptr), unsafe.Pointer(&d.Ptr), unsafe.Pointer(&m.Raw.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&kk), unsafe.Pointer(&nn), unsafe.Pointer(&bb))
}

func gemmQ6KBatch8ToBuffer(out, x *Buffer, batch int, m *GPUQKMatrix) error {
	if batch <= 0 || m == nil || m.kind != qkKindQ6K || m.Raw == nil || x == nil || out == nil || m.Raw.Ptr == 0 || x.Ptr == 0 || out.Ptr == 0 || x.Size < batch*m.InDim*4 || out.Size < batch*m.OutDim*4 {
		return fmt.Errorf("invalid Q6_K batch8 GEMM buffers")
	}
	kk, nn, bb := uint32(m.InDim), uint32(m.OutDim), uint32(batch)
	return LaunchKernel(fnQ6KGemmBatch8, uint32((m.OutDim+3)/4), uint32((batch+7)/8), 1, 128, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&m.Raw.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&kk), unsafe.Pointer(&nn), unsafe.Pointer(&bb))
}

func gemmQ6KWarpToBuffer(out, x *Buffer, batch int, m *GPUQKMatrix) error {
	if batch <= 0 || m == nil || m.kind != qkKindQ6K || m.Raw == nil || x == nil || out == nil || m.Raw.Ptr == 0 || x.Ptr == 0 || out.Ptr == 0 || x.Size < batch*m.InDim*4 || out.Size < batch*m.OutDim*4 {
		return fmt.Errorf("invalid Q6_K warp GEMM buffers")
	}
	kk, nn, bb := uint32(m.InDim), uint32(m.OutDim), uint32(batch)
	return LaunchKernel(fnQ6KGemmWarp, uint32((m.OutDim+3)/4), bb, 1, 128, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&m.Raw.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&kk), unsafe.Pointer(&nn), unsafe.Pointer(&bb))
}

func gemvQKBatchToBuffer(out, x *Buffer, batch int, m *GPUQKMatrix, kind uint8, fn CUfunction) error {
	if batch <= 0 {
		return fmt.Errorf("Q%d_K batch=%d must be positive", kind, batch)
	}
	if m == nil || m.kind != kind || m.Raw == nil || m.Raw.Ptr == 0 || x == nil || out == nil || x.Ptr == 0 || out.Ptr == 0 || fn == 0 || x.Size < batch*m.InDim*4 || out.Size < batch*m.OutDim*4 {
		return fmt.Errorf("invalid Q%d_K batch GEMV buffers", kind)
	}
	inDim, outDim, batchU := uint32(m.InDim), uint32(m.OutDim), uint32(batch)
	return LaunchKernel(fn, uint32(m.OutDim), uint32(batch), 1, 256, 1, 1, 0,
		unsafe.Pointer(&x.Ptr), unsafe.Pointer(&m.Raw.Ptr), unsafe.Pointer(&out.Ptr),
		unsafe.Pointer(&inDim), unsafe.Pointer(&outDim), unsafe.Pointer(&batchU))
}

func GemvQ5KBatch(out, x []float32, batch int, m *GPUQKMatrix) error {
	return gemvQKBatch(out, x, batch, m, qkKindQ5K)
}

func GemvQ6KBatch(out, x []float32, batch int, m *GPUQKMatrix) error {
	return gemvQKBatch(out, x, batch, m, qkKindQ6K)
}

func gemvQKBatch(out, x []float32, batch int, m *GPUQKMatrix, kind uint8) error {
	if m == nil || batch <= 0 || len(x) < batch*m.InDim || len(out) < batch*m.OutDim {
		return fmt.Errorf("invalid Q%d_K host batch GEMV buffers", kind)
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
	if kind == qkKindQ5K {
		err = GemvQ5KBatchToBuffer(outBuf, xBuf, batch, m)
	} else {
		err = GemvQ6KBatchToBuffer(outBuf, xBuf, batch, m)
	}
	if err != nil {
		return err
	}
	return outBuf.Download(out[:batch*m.OutDim])
}
