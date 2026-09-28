package nvidia

import (
	"fmt"
	"unsafe"

	"github.com/rcarmo/go-pherence/loader/gguf"
)

// GPUGGUFMatrix is a resident projection in one of the Go System One-admitted K formats.
type GPUGGUFMatrix struct {
	QType  gguf.QuantType
	InDim  int
	OutDim int
	q4k    *GPUQ4KMatrix
	qk     *GPUQKMatrix
}

func UploadGGUFMatrix(m *gguf.QuantMatrix) (*GPUGGUFMatrix, error) {
	if m == nil || m.InDim <= 0 || m.OutDim <= 0 {
		return nil, fmt.Errorf("invalid GGUF matrix")
	}
	out := &GPUGGUFMatrix{QType: m.QType, InDim: m.InDim, OutDim: m.OutDim}
	var err error
	switch m.QType {
	case gguf.QuantQ4_K:
		out.q4k, err = UploadQ4KMatrixRaw(m.Raw, m.InDim, m.OutDim)
	case gguf.QuantQ5_K:
		out.qk, err = UploadQ5KMatrixRows(m.Raw, m.InDim, m.OutDim)
	case gguf.QuantQ6_K:
		out.qk, err = UploadQ6KMatrixRows(m.Raw, m.InDim, m.OutDim)
	default:
		return nil, fmt.Errorf("GGUF GPU matrix %s type=%s unsupported", m.Name, m.QType)
	}
	if err != nil {
		return nil, fmt.Errorf("upload GGUF GPU matrix %s: %w", m.Name, err)
	}
	return out, nil
}

func (m *GPUGGUFMatrix) ResidentBytes() int {
	if m == nil {
		return 0
	}
	if m.q4k != nil {
		n := 0
		for _, b := range []*Buffer{m.q4k.Q, m.q4k.Scales, m.q4k.Mins} {
			if b != nil {
				n += b.Size
			}
		}
		return n
	}
	if m.qk != nil {
		n := 0
		for _, b := range []*Buffer{m.qk.Raw, m.qk.PackedQ, m.qk.PackedScale, m.qk.PackedMin} {
			if b != nil {
				n += b.Size
			}
		}
		return n
	}
	return 0
}

func (m *GPUGGUFMatrix) Free() {
	if m == nil {
		return
	}
	if m.q4k != nil {
		m.q4k.Free()
		m.q4k = nil
	}
	if m.qk != nil {
		m.qk.Free()
		m.qk = nil
	}
}

// ProjectQ4PairToBuffers quantises x once and projects it through two Q4_K
// matrices with identical input dimensions. Output dimensions may differ.
func ProjectQ4PairToBuffers(outA, outB, x *Buffer, batch int, a, b *GPUGGUFMatrix) error {
	if a != nil && b != nil && a.q4k != nil && b.q4k != nil && a.q4k.raw && b.q4k.raw && a.InDim == b.InDim {
		return gemmQ4RawUpstreamPair(outA, outB, x, batch, a.q4k, b.q4k)
	}
	if a == nil || b == nil || a.QType != gguf.QuantQ4_K || b.QType != gguf.QuantQ4_K || a.q4k == nil || b.q4k == nil || !a.q4k.coalesced || !b.q4k.coalesced || a.InDim != b.InDim {
		if a == nil || b == nil {
			return fmt.Errorf("invalid Q4_K projection pair")
		}
		if err := a.ProjectBatchToBuffer(outA, x, batch); err != nil {
			return err
		}
		return b.ProjectBatchToBuffer(outB, x, batch)
	}
	groups := a.InDim / 32
	q, d, s, unlock, err := q8ProjectionBuffers(batch*a.InDim, batch*groups, batch*groups*4)
	if err != nil {
		return err
	}
	defer unlock()
	if err = quantizeQ8RowsSum(q, d, s, x, batch, a.InDim); err != nil {
		return err
	}
	if batch >= 16 && fnQ4PairMMQJ16 != 0 {
		return gemmQ4PairPrepared(outA, outB, q, d, s, batch, a.q4k, b.q4k)
	}
	if err = gemmQ4CoalescedPrepared(outA, q, d, s, batch, a.q4k); err != nil {
		return err
	}
	return gemmQ4CoalescedPrepared(outB, q, d, s, batch, b.q4k)
}

func (m *GPUGGUFMatrix) ProjectSelectedRows(out, x, rows *Buffer, count int) error {
	if m == nil || m.QType != gguf.QuantQ5_K || m.qk == nil || m.qk.PackedQ == nil || m.qk.PackedScale == nil || m.qk.PackedMin == nil || out == nil || x == nil || rows == nil || count <= 0 || out.Size < count*4 || x.Size < m.InDim*4 || rows.Size < count*4 || fnQ5PackedSelected == 0 {
		return fmt.Errorf("invalid selected Q5_K projection")
	}
	kk, cc := uint32(m.InDim), uint32(count)
	return LaunchKernel(fnQ5PackedSelected, uint32(count), 1, 1, 32, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&m.qk.PackedQ.Ptr), unsafe.Pointer(&m.qk.PackedScale.Ptr), unsafe.Pointer(&m.qk.PackedMin.Ptr), unsafe.Pointer(&rows.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&kk), unsafe.Pointer(&cc))
}

func (m *GPUGGUFMatrix) ProjectBatchToBuffer(out, x *Buffer, batch int) error {
	if m == nil || batch <= 0 || out == nil || x == nil || out.Size < batch*m.OutDim*4 || x.Size < batch*m.InDim*4 {
		return fmt.Errorf("invalid GGUF GPU projection batch=%d", batch)
	}
	switch m.QType {
	case gguf.QuantQ4_K:
		return GemvQ4KBatchToBuffer(out, x, batch, m.q4k)
	case gguf.QuantQ5_K:
		return GemvQ5KBatchToBuffer(out, x, batch, m.qk)
	case gguf.QuantQ6_K:
		return GemvQ6KBatchToBuffer(out, x, batch, m.qk)
	default:
		return fmt.Errorf("GGUF GPU projection type=%s unsupported", m.QType)
	}
}
