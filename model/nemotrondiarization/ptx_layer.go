package nemotrondiarization

import (
	"fmt"
	"math"
	"sync"

	ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
)

// PTXAudioLayer retains one diarization transformer layer's immutable F32
// weights on CUDA. The caller supplies exact, unmasked rows with RoPE positions
// starting at zero. ForwardBuffer serializes Close and each layer launch; the
// output must not overlap the input. It does not run the first layer or head.
type PTXAudioLayer struct {
	mu                                     sync.Mutex
	closed                                 bool
	maxRows                                int
	owned                                  []*ptx.Buffer
	frequency                              *ptx.Buffer
	qW, kW, vW, oW, oB, n1W, n1B, n2W, n2B *ptx.Buffer
	fc1W, fc1B, fc2W, fc2B                 *ptx.Buffer
}

func NewPTXAudioLayer(source *Layer1Complete, maxRows int) (layer *PTXAudioLayer, err error) {
	if source == nil || source.attention == nil || source.attention.qkv == nil ||
		maxRows < 1 || maxRows > maxPreparedDiarizationRows || !ptx.SgemmReady() {
		return nil, fmt.Errorf("invalid PTX diarization layer model or row capacity")
	}
	attn, qkv := source.attention, source.attention.qkv
	const width, hidden = projectedWidth, diarizationIntermediate
	for _, item := range []struct {
		values []float32
		size   int
	}{
		{qkv.gamma, width}, {qkv.beta, width}, {qkv.q, width * width}, {qkv.k, width * width}, {qkv.v, width * width},
		{attn.outWeight, width * width}, {attn.outBias, width},
		{source.normWeight, width}, {source.normBias, width}, {source.fc1Weight, hidden * width},
		{source.fc1Bias, hidden}, {source.fc2Weight, width * hidden}, {source.fc2Bias, width},
	} {
		if len(item.values) != item.size {
			return nil, fmt.Errorf("invalid PTX diarization layer weight shape")
		}
		for _, value := range item.values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("non-finite PTX diarization layer weight")
			}
		}
	}
	candidate := &PTXAudioLayer{maxRows: maxRows}
	defer func() {
		if err != nil {
			candidate.Close()
		}
	}()
	upload := func(data []float32) (*ptx.Buffer, error) {
		b, e := ptx.Malloc(len(data))
		if e != nil {
			return nil, e
		}
		candidate.owned = append(candidate.owned, b)
		if e = b.Upload(data); e != nil {
			return nil, e
		}
		return b, nil
	}
	transpose := func(w []float32, n, k int) []float32 {
		out := make([]float32, len(w))
		for j := 0; j < n; j++ {
			for i := 0; i < k; i++ {
				out[i*n+j] = w[j*k+i]
			}
		}
		return out
	}
	for _, item := range []struct {
		dst  **ptx.Buffer
		data []float32
	}{
		{&candidate.qW, transpose(qkv.q, width, width)}, {&candidate.kW, transpose(qkv.k, width, width)},
		{&candidate.vW, transpose(qkv.v, width, width)}, {&candidate.oW, transpose(attn.outWeight, width, width)},
		{&candidate.oB, attn.outBias}, {&candidate.n1W, qkv.gamma}, {&candidate.n1B, qkv.beta},
		{&candidate.n2W, source.normWeight}, {&candidate.n2B, source.normBias},
		{&candidate.fc1W, transpose(source.fc1Weight, hidden, width)}, {&candidate.fc1B, source.fc1Bias},
		{&candidate.fc2W, transpose(source.fc2Weight, width, hidden)}, {&candidate.fc2B, source.fc2Bias},
	} {
		*item.dst, err = upload(item.data)
		if err != nil {
			return nil, err
		}
	}
	const half = diarizationHeadWidth / 2
	freqs := make([]float32, maxRows*half*2)
	for row := 0; row < maxRows; row++ {
		for d := 0; d < half; d++ {
			angle := float32(row) * float32(1/math.Pow(10000, float64(2*d)/diarizationHeadWidth))
			freqs[(row*half+d)*2] = float32(math.Cos(float64(angle)))
			freqs[(row*half+d)*2+1] = float32(math.Sin(float64(angle)))
		}
	}
	candidate.frequency, err = upload(freqs)
	if err != nil {
		return nil, err
	}
	return candidate, nil
}

// Close is idempotent; it waits for any current ForwardBuffer call.
func (l *PTXAudioLayer) Close() {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return
	}
	l.closed = true
	for i := len(l.owned) - 1; i >= 0; i-- {
		l.owned[i].Free()
	}
	l.owned = nil
}

// Forward returns a caller-owned host output and preserves the host input.
func (l *PTXAudioLayer) Forward(input []float32, rows int) ([]float32, error) {
	if l == nil || rows < 1 || rows > l.maxRows || len(input) != rows*projectedWidth {
		return nil, fmt.Errorf("invalid PTX diarization layer input")
	}
	for _, value := range input {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite PTX diarization layer input")
		}
	}
	in, err := ptx.Malloc(len(input))
	if err != nil {
		return nil, err
	}
	defer in.Free()
	out, err := ptx.Malloc(len(input))
	if err != nil {
		return nil, err
	}
	defer out.Free()
	if err := in.Upload(input); err != nil {
		return nil, err
	}
	if err := l.ForwardBuffer(out, in, rows); err != nil {
		return nil, err
	}
	result := make([]float32, len(input))
	if err := out.Download(result); err != nil {
		return nil, err
	}
	return result, nil
}

func deviceBuffersOverlap(a, b *ptx.Buffer, bytes uint64) bool {
	a0, b0 := uint64(a.Ptr), uint64(b.Ptr)
	if a0+bytes < a0 || b0+bytes < b0 {
		return true
	}
	return a0 < b0+bytes && b0 < a0+bytes
}

// ForwardBuffer keeps every intermediate resident; it synchronizes before
// releasing per-call scratch or permitting Close. No CPU fallback is used.
func (l *PTXAudioLayer) ForwardBuffer(out, in *ptx.Buffer, rows int) (err error) {
	if l == nil || rows < 1 || rows > l.maxRows || in == nil || out == nil || in.Ptr == 0 || out.Ptr == 0 {
		return fmt.Errorf("invalid PTX diarization layer device input")
	}
	n, ok := checked.MulInt(rows, projectedWidth)
	bytes, okBytes := checked.MulInt(n, 4)
	if !ok || !okBytes || in.Size < bytes || out.Size < bytes ||
		deviceBuffersOverlap(in, out, uint64(bytes)) {
		return fmt.Errorf("invalid or overlapping PTX diarization layer device input")
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return fmt.Errorf("PTX diarization layer closed")
	}
	var scratch []*ptx.Buffer
	defer func() {
		// CUDA work may be asynchronous even when a later launch fails.
		if syncErr := ptx.SyncErr(); err == nil {
			err = syncErr
		}
		for i := len(scratch) - 1; i >= 0; i-- {
			scratch[i].Free()
		}
	}()
	alloc := func(n int) (*ptx.Buffer, error) {
		b, e := ptx.Malloc(n)
		if e == nil {
			scratch = append(scratch, b)
		}
		return b, e
	}
	widthN, intermediateN := rows*projectedWidth, rows*diarizationIntermediate
	var norm, q, k, v, mixed, projected, residual, norm2, fc1 *ptx.Buffer
	for _, item := range []struct {
		dst  **ptx.Buffer
		size int
	}{
		{&norm, widthN}, {&q, widthN}, {&k, widthN}, {&v, widthN}, {&mixed, widthN},
		{&projected, widthN}, {&residual, widthN}, {&norm2, widthN}, {&fc1, intermediateN},
	} {
		*item.dst, err = alloc(item.size)
		if err != nil {
			return err
		}
	}
	if err = ptx.AffineLayerNormF32Buffer(norm, in, l.n1W, l.n1B, rows, projectedWidth, 1e-5); err != nil {
		return err
	}
	for _, op := range []struct{ dst, weight *ptx.Buffer }{{q, l.qW}, {k, l.kW}, {v, l.vW}} {
		if err = ptx.Sgemm(rows, projectedWidth, projectedWidth, 1, norm, op.weight, op.dst); err != nil {
			return err
		}
	}
	for _, tensor := range []*ptx.Buffer{q, k} {
		if err = ptx.RoPEPartialSequenceBuffer(tensor, l.frequency, rows, 0, diarizationHeads, diarizationHeadWidth, diarizationHeadWidth/2); err != nil {
			return err
		}
	}
	if err = ptx.WhisperAttentionFullOnlineBuffer(mixed, q, k, v, rows, rows, diarizationHeads, diarizationHeadWidth, 1.0/8.0); err != nil {
		return err
	}
	if err = ptx.Sgemm(rows, projectedWidth, projectedWidth, 1, mixed, l.oW, projected); err != nil {
		return err
	}
	if err = ptx.WhisperRowBiasBuffer(projected, l.oB, rows, projectedWidth); err != nil {
		return err
	}
	if err = ptx.VecAddF32Buffer(in, projected, residual, widthN); err != nil {
		return err
	}
	if err = ptx.AffineLayerNormF32Buffer(norm2, residual, l.n2W, l.n2B, rows, projectedWidth, 1e-5); err != nil {
		return err
	}
	if err = ptx.Sgemm(rows, diarizationIntermediate, projectedWidth, 1, norm2, l.fc1W, fc1); err != nil {
		return err
	}
	if err = ptx.WhisperRowBiasBuffer(fc1, l.fc1B, rows, diarizationIntermediate); err != nil {
		return err
	}
	if err = ptx.GELUErfF32Buffer(fc1, intermediateN); err != nil {
		return err
	}
	if err = ptx.Sgemm(rows, projectedWidth, diarizationIntermediate, 1, fc1, l.fc2W, out); err != nil {
		return err
	}
	if err = ptx.WhisperRowBiasBuffer(out, l.fc2B, rows, projectedWidth); err != nil {
		return err
	}
	return ptx.VecAddF32Buffer(residual, out, out, widthN)
}
