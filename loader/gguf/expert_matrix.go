package gguf

import (
	"fmt"
	"runtime"
	"sync"
)

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
	if m == nil || expert < 0 || expert >= m.Experts || len(out) < m.OutDim || len(x) < m.InDim {
		return fmt.Errorf("gguf expert gemv %s: bad buffers", m.Name)
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	if m.OutDim < workers*8 {
		workers = 1
	}
	if workers == 1 {
		return m.gemvExpertRange(out, x, expert, 0, m.OutDim)
	}
	chunk := (m.OutDim + workers - 1) / workers
	var wg sync.WaitGroup
	errCh := make(chan error, workers)
	for wid := 0; wid < workers; wid++ {
		start := wid * chunk
		end := start + chunk
		if end > m.OutDim {
			end = m.OutDim
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			if err := m.gemvExpertRange(out, x, expert, start, end); err != nil {
				errCh <- err
			}
		}(start, end)
	}
	wg.Wait()
	select {
	case err := <-errCh:
		return err
	default:
		return nil
	}
}

func (m *ExpertMatrices) gemvExpertRange(out, x []float32, expert, start, end int) error {
	row := make([]float32, m.InDim)
	for r := start; r < end; r++ {
		if err := m.DequantExpertRowTo(row, expert, r); err != nil {
			return err
		}
		out[r] = dotExpertRow(row, x[:m.InDim])
	}
	return nil
}

func dotExpertRow(a, b []float32) float32 {
	var sum float32
	for i, av := range a {
		sum += av * b[i]
	}
	return sum
}
