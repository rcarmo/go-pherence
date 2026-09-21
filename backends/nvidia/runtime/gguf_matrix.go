package nvidia

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/gguf"
)

// GPUGGUFMatrix is a resident projection in one of the QEV-admitted K formats.
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
		out.q4k, err = UploadQ4KMatrixRows(m.Raw, m.InDim, m.OutDim)
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
