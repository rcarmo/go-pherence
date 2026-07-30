package model

import (
	"fmt"
	"sort"
	"unsafe"
)

const PrefillChunkBudgetUnlimited = -1

var defaultPrefillChunkSizes = []int{32, 64, 128}

const (
	prefillCPUHiddenBuffersPerRow = 6 // hidden + residual + normed + mlpIn + oOut + down
	prefillGPUHiddenBuffersPerRow = 5 // hidden + residual + normed + oOut + down
	prefillQBuffersPerRow         = 2 // q + attnOut
	prefillKVBuffersPerRow        = 2 // k + v
	prefillMLPBuffersPerRow       = 2 // gate + up
	prefillCPUReusableBufferCount = 12
	prefillGPUReusableBufferCount = 11
)

var (
	prefillFloat32Bytes     = int(unsafe.Sizeof(float32(0)))
	prefillSliceHeaderBytes = int(unsafe.Sizeof([]float32(nil)))
)

// PrefillChunkModelDims describes the reusable batch-buffer widths for prompt
// prefill. Layers are validated and carried through the plan even though the
// current scratch buffers are reused layer-to-layer.
type PrefillChunkModelDims struct {
	HiddenSize   int
	QDim         int
	KVDim        int
	Intermediate int
	Layers       int
}

// PrefillChunkScratchEstimate conservatively models the reusable scratch needed
// by the current CPU/GPU prefill implementations.
type PrefillChunkScratchEstimate struct {
	Dims             PrefillChunkModelDims
	CPURowBytes      int
	GPURowBytes      int
	ReusableRowBytes int
	FixedBytes       int
}

// TotalBytes estimates the peak reusable scratch for rows prompt tokens.
func (e PrefillChunkScratchEstimate) TotalBytes(rows int) (int, error) {
	if rows < 0 {
		return 0, fmt.Errorf("prefill chunk rows %d out of range", rows)
	}
	if rows == 0 {
		return 0, nil
	}
	rowBytes, ok := checkedProduct(rows, e.ReusableRowBytes)
	if !ok {
		return 0, fmt.Errorf("prefill chunk scratch byte size overflow rows=%d per_row=%d", rows, e.ReusableRowBytes)
	}
	total, ok := checkedAddNonNegative(e.FixedBytes, rowBytes)
	if !ok {
		return 0, fmt.Errorf("prefill chunk scratch total overflow rows=%d fixed=%d per_row=%d", rows, e.FixedBytes, e.ReusableRowBytes)
	}
	return total, nil
}

// PrefillChunkSpan is one absolute prompt-token range [Start, End) to prefill.
type PrefillChunkSpan struct {
	Start int
	End   int
}

// PrefillChunkPlan is a backend-neutral chunking plan for prompt prefill.
type PrefillChunkPlan struct {
	Tokens            int
	Dims              PrefillChunkModelDims
	AllowedChunkSizes []int
	BudgetBytes       int
	ChunkSize         int
	Disabled          bool
	Spans             []PrefillChunkSpan
	Scratch           PrefillChunkScratchEstimate
	PeakRows          int
	PeakScratchBytes  int
}

// StructSpans converts the plan's absolute spans to a plain Start/End struct
// slice for backends that accept span coordinates without importing model.
func (p PrefillChunkPlan) StructSpans() []struct {
	Start int
	End   int
} {
	if len(p.Spans) == 0 {
		return nil
	}
	out := make([]struct {
		Start int
		End   int
	}, len(p.Spans))
	for i, span := range p.Spans {
		out[i] = struct {
			Start int
			End   int
		}{Start: span.Start, End: span.End}
	}
	return out
}

// EstimatePrefillChunkScratch derives a conservative reusable scratch estimate
// from the current CPU/GPU prefill batch buffers.
func EstimatePrefillChunkScratch(dims PrefillChunkModelDims) (PrefillChunkScratchEstimate, error) {
	if err := validatePrefillChunkModelDims(dims); err != nil {
		return PrefillChunkScratchEstimate{}, err
	}
	cpuHidden, okCPUHidden := checkedProduct(dims.HiddenSize, prefillCPUHiddenBuffersPerRow)
	gpuHidden, okGPUHidden := checkedProduct(dims.HiddenSize, prefillGPUHiddenBuffersPerRow)
	qFloats, okQ := checkedProduct(dims.QDim, prefillQBuffersPerRow)
	kvFloats, okKV := checkedProduct(dims.KVDim, prefillKVBuffersPerRow)
	mlpFloats, okMLP := checkedProduct(dims.Intermediate, prefillMLPBuffersPerRow)
	if !okCPUHidden || !okGPUHidden || !okQ || !okKV || !okMLP {
		return PrefillChunkScratchEstimate{}, fmt.Errorf("prefill chunk scratch row size overflow dims=%+v", dims)
	}
	cpuRowFloats, ok := checkedAddNonNegative(cpuHidden, qFloats)
	if ok {
		cpuRowFloats, ok = checkedAddNonNegative(cpuRowFloats, kvFloats)
	}
	if ok {
		cpuRowFloats, ok = checkedAddNonNegative(cpuRowFloats, mlpFloats)
	}
	if !ok {
		return PrefillChunkScratchEstimate{}, fmt.Errorf("prefill chunk CPU scratch row total overflow dims=%+v", dims)
	}
	gpuRowFloats, ok := checkedAddNonNegative(gpuHidden, qFloats)
	if ok {
		gpuRowFloats, ok = checkedAddNonNegative(gpuRowFloats, kvFloats)
	}
	if ok {
		gpuRowFloats, ok = checkedAddNonNegative(gpuRowFloats, mlpFloats)
	}
	if !ok {
		return PrefillChunkScratchEstimate{}, fmt.Errorf("prefill chunk GPU scratch row total overflow dims=%+v", dims)
	}
	cpuRowBytes, okCPUBytes := checkedProduct(cpuRowFloats, prefillFloat32Bytes)
	gpuRowBytes, okGPUBytes := checkedProduct(gpuRowFloats, prefillFloat32Bytes)
	cpuFixedBytes, okCPUFixed := checkedProduct(prefillCPUReusableBufferCount, prefillSliceHeaderBytes)
	gpuFixedBytes, okGPUFixed := checkedProduct(prefillGPUReusableBufferCount, prefillSliceHeaderBytes)
	if !okCPUBytes || !okGPUBytes || !okCPUFixed || !okGPUFixed {
		return PrefillChunkScratchEstimate{}, fmt.Errorf("prefill chunk scratch byte conversion overflow dims=%+v", dims)
	}
	fixedBytes := cpuFixedBytes
	if gpuFixedBytes > fixedBytes {
		fixedBytes = gpuFixedBytes
	}
	reusableRowBytes := cpuRowBytes
	if gpuRowBytes > reusableRowBytes {
		reusableRowBytes = gpuRowBytes
	}
	return PrefillChunkScratchEstimate{
		Dims:             dims,
		CPURowBytes:      cpuRowBytes,
		GPURowBytes:      gpuRowBytes,
		ReusableRowBytes: reusableRowBytes,
		FixedBytes:       fixedBytes,
	}, nil
}

// NewPrefillChunkPlan selects the largest allowed chunk size that fits the
// reusable scratch budget and emits absolute prompt-token spans. Passing
// PrefillChunkBudgetUnlimited disables chunking and returns a single full-batch
// span.
func NewPrefillChunkPlan(tokens int, dims PrefillChunkModelDims, availableScratchBytes int, allowedChunkSizes []int) (PrefillChunkPlan, error) {
	if tokens < 0 {
		return PrefillChunkPlan{}, fmt.Errorf("prefill chunk token count %d out of range", tokens)
	}
	if availableScratchBytes < PrefillChunkBudgetUnlimited {
		return PrefillChunkPlan{}, fmt.Errorf("prefill chunk scratch budget %d out of range", availableScratchBytes)
	}
	estimate, err := EstimatePrefillChunkScratch(dims)
	if err != nil {
		return PrefillChunkPlan{}, err
	}
	normalized, err := normalizePrefillChunkSizes(allowedChunkSizes)
	if err != nil {
		return PrefillChunkPlan{}, err
	}
	plan := PrefillChunkPlan{
		Tokens:            tokens,
		Dims:              dims,
		AllowedChunkSizes: normalized,
		BudgetBytes:       availableScratchBytes,
		Scratch:           estimate,
	}
	if tokens == 0 {
		return plan, nil
	}
	if availableScratchBytes == PrefillChunkBudgetUnlimited {
		plan.Disabled = true
		plan.ChunkSize = tokens
		plan.Spans, err = buildPrefillChunkSpans(tokens, tokens)
		if err != nil {
			return PrefillChunkPlan{}, err
		}
		plan.PeakRows = tokens
		plan.PeakScratchBytes, err = estimate.TotalBytes(tokens)
		if err != nil {
			return PrefillChunkPlan{}, err
		}
		return plan, nil
	}
	chosen := 0
	if tokens < normalized[0] {
		need, err := estimate.TotalBytes(tokens)
		if err != nil {
			return PrefillChunkPlan{}, err
		}
		if need > availableScratchBytes {
			return PrefillChunkPlan{}, fmt.Errorf("prefill chunk scratch budget %d too small for tail batch of %d tokens (need %d)", availableScratchBytes, tokens, need)
		}
		chosen = tokens
	} else {
		for i := len(normalized) - 1; i >= 0; i-- {
			size := normalized[i]
			if size > tokens {
				continue
			}
			need, err := estimate.TotalBytes(size)
			if err != nil {
				return PrefillChunkPlan{}, err
			}
			if need <= availableScratchBytes {
				chosen = size
				break
			}
		}
		if chosen == 0 {
			return PrefillChunkPlan{}, fmt.Errorf("prefill chunk scratch budget %d too small for allowed chunk sizes %v", availableScratchBytes, normalized)
		}
	}
	plan.ChunkSize = chosen
	plan.Spans, err = buildPrefillChunkSpans(tokens, chosen)
	if err != nil {
		return PrefillChunkPlan{}, err
	}
	for _, span := range plan.Spans {
		rows := span.End - span.Start
		if rows > plan.PeakRows {
			plan.PeakRows = rows
		}
	}
	plan.PeakScratchBytes, err = estimate.TotalBytes(plan.PeakRows)
	if err != nil {
		return PrefillChunkPlan{}, err
	}
	return plan, nil
}

func validatePrefillChunkModelDims(dims PrefillChunkModelDims) error {
	if dims.HiddenSize <= 0 {
		return fmt.Errorf("prefill chunk hidden size %d out of range", dims.HiddenSize)
	}
	if dims.QDim <= 0 {
		return fmt.Errorf("prefill chunk q dim %d out of range", dims.QDim)
	}
	if dims.KVDim <= 0 {
		return fmt.Errorf("prefill chunk kv dim %d out of range", dims.KVDim)
	}
	if dims.Intermediate <= 0 {
		return fmt.Errorf("prefill chunk intermediate size %d out of range", dims.Intermediate)
	}
	if dims.Layers <= 0 {
		return fmt.Errorf("prefill chunk layer count %d out of range", dims.Layers)
	}
	return nil
}

func normalizePrefillChunkSizes(sizes []int) ([]int, error) {
	if len(sizes) == 0 {
		return append([]int(nil), defaultPrefillChunkSizes...), nil
	}
	norm := append([]int(nil), sizes...)
	sort.Ints(norm)
	out := norm[:0]
	last := 0
	for i, size := range norm {
		if size <= 0 {
			return nil, fmt.Errorf("prefill chunk size %d out of range", size)
		}
		if i > 0 && size == last {
			continue
		}
		out = append(out, size)
		last = size
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("prefill chunk sizes empty")
	}
	return append([]int(nil), out...), nil
}

func buildPrefillChunkSpans(tokens, chunkSize int) ([]PrefillChunkSpan, error) {
	if tokens < 0 {
		return nil, fmt.Errorf("prefill chunk token count %d out of range", tokens)
	}
	if tokens == 0 {
		return nil, nil
	}
	if chunkSize <= 0 {
		return nil, fmt.Errorf("prefill chunk size %d out of range", chunkSize)
	}
	spans := make([]PrefillChunkSpan, 0)
	for start := 0; start < tokens; {
		end, ok := checkedAddNonNegative(start, chunkSize)
		if !ok {
			return nil, fmt.Errorf("prefill chunk span overflow start=%d size=%d", start, chunkSize)
		}
		if end > tokens {
			end = tokens
		}
		spans = append(spans, PrefillChunkSpan{Start: start, End: end})
		start = end
	}
	return spans, nil
}
