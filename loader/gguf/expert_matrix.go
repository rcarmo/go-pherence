package gguf

import "fmt"

// ExpertMatrices keeps a GGUF 3D MoE expert tensor in its original quantized
// form. GGUF MoE tensors are expected as [inDim, outDim, experts], making each
// expert a contiguous stack of output rows.
type ExpertMatrices struct {
	Name    string
	QType   QuantType
	Raw     []byte
	InDim   int
	OutDim  int
	Experts int
}

func (g *GGUF) ExpertMatricesFromTensor(t TensorInfo) (*ExpertMatrices, error) {
	if len(t.Shape) != 3 {
		return nil, fmt.Errorf("gguf: tensor %q shape %v is not an expert tensor", t.Name, t.Shape)
	}
	raw, err := g.Raw(t)
	if err != nil {
		return nil, err
	}
	m := &ExpertMatrices{Name: t.Name, QType: t.QType, Raw: raw, InDim: int(t.Shape[0]), OutDim: int(t.Shape[1]), Experts: int(t.Shape[2])}
	rowBytes, err := m.RowBytes()
	if err != nil {
		return nil, err
	}
	need := rowBytes * m.OutDim * m.Experts
	if need < 0 || len(raw) < need {
		return nil, fmt.Errorf("gguf: expert tensor %q raw short need=%d have=%d", t.Name, need, len(raw))
	}
	return m, nil
}

func (m *ExpertMatrices) RowBytes() (int, error) { return TensorRawBytes(m.QType, m.InDim) }

func (m *ExpertMatrices) DequantExpertRowTo(dst []float32, expert, row int) error {
	if m == nil || expert < 0 || expert >= m.Experts || row < 0 || row >= m.OutDim || len(dst) < m.InDim {
		return fmt.Errorf("gguf expert row %s: bad expert=%d row=%d dst=%d dims=[%d,%d,%d]", m.Name, expert, row, len(dst), m.InDim, m.OutDim, m.Experts)
	}
	rowBytes, err := m.RowBytes()
	if err != nil {
		return err
	}
	rowIndex := expert*m.OutDim + row
	start := rowIndex * rowBytes
	end := start + rowBytes
	if end > len(m.Raw) {
		return fmt.Errorf("gguf expert row %s: expert=%d row=%d raw short", m.Name, expert, row)
	}
	return dequantRowTo(dst[:m.InDim], m.Raw[start:end], m.QType, m.InDim)
}

func (m *ExpertMatrices) GemvExpertTo(out, x []float32, expert int) error {
	if m == nil || len(out) < m.OutDim || len(x) < m.InDim {
		return fmt.Errorf("gguf expert gemv %s: bad buffers", m.Name)
	}
	row := make([]float32, m.InDim)
	for r := 0; r < m.OutDim; r++ {
		if err := m.DequantExpertRowTo(row, expert, r); err != nil {
			return err
		}
		var sum float32
		for i := 0; i < m.InDim; i++ {
			sum += row[i] * x[i]
		}
		out[r] = sum
	}
	return nil
}
