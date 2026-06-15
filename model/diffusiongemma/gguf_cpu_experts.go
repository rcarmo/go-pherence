package diffusiongemma

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/internal/checked"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

// GGUFExpertIndex holds the GGUF expert matrices for all layers, using the
// same quantized weight format as llama.cpp (Q4_K for gate_up, mixed K/Q formats for down).
// This matches the ggml_mul_mat_id strategy exactly.
type GGUFExpertIndex struct {
	NumLayers    int
	NumExperts   int
	Intermediate int // 704 for DiffusionGemma
	HiddenSize   int // 2816
	entries      []ggufLayerExperts
}

type ggufLayerExperts struct {
	// gateUp is [InDim=2816, OutDim=1408, Experts=128] in Q4_K
	// OutDim=1408 = 704(gate) + 704(up) fused
	gateUp *gguf.ExpertMatrices
	// down is [InDim=704, OutDim=2816, Experts=128] in the GGUF tensor's quant type
	// (Q8_0/Q5_0 in the Q4_K_M DiffusionGemma checkpoint).
	down *gguf.ExpertMatrices
	// downScale is per-expert scale for down projection (optional)
	downScale []float32
}

// BuildGGUFExpertIndex loads expert matrices from a GGUF file.
// Tensor naming follows llama.cpp convention:
//
//	blk.{L}.ffn_gate_up_exps.weight  — fused gate+up [2816, 1408, 128] Q4_K
//	blk.{L}.ffn_down_exps.weight     — down [704, 2816, 128] GGUF quant (e.g. Q8_0/Q5_0)
//	blk.{L}.ffn_down_exps.scale      — per-expert scale [128] F32
func BuildGGUFExpertIndex(g *gguf.GGUF, numLayers, numExperts int) (*GGUFExpertIndex, error) {
	t0 := time.Now()
	idx := &GGUFExpertIndex{
		NumLayers:  numLayers,
		NumExperts: numExperts,
		entries:    make([]ggufLayerExperts, numLayers),
	}

	for l := 0; l < numLayers; l++ {
		guName := fmt.Sprintf("blk.%d.ffn_gate_up_exps.weight", l)
		guTensor, ok := g.TensorByName(guName)
		if !ok {
			return nil, fmt.Errorf("GGUF expert index: tensor %q not found", guName)
		}
		guMat, err := g.ExpertMatricesFromTensor(guTensor)
		if err != nil {
			return nil, fmt.Errorf("GGUF expert index layer %d gate_up: %w", l, err)
		}

		dnName := fmt.Sprintf("blk.%d.ffn_down_exps.weight", l)
		dnTensor, ok := g.TensorByName(dnName)
		if !ok {
			return nil, fmt.Errorf("GGUF expert index: tensor %q not found", dnName)
		}
		dnMat, err := g.ExpertMatricesFromTensor(dnTensor)
		if err != nil {
			return nil, fmt.Errorf("GGUF expert index layer %d down: %w", l, err)
		}

		var downScale []float32
		dsName := fmt.Sprintf("blk.%d.ffn_down_exps.scale", l)
		if dsTensor, ok := g.TensorByName(dsName); ok {
			raw, err := g.Raw(dsTensor)
			if err != nil {
				return nil, fmt.Errorf("GGUF expert index layer %d down scale: %w", l, err)
			}
			if len(raw) < numExperts*4 {
				return nil, fmt.Errorf("GGUF expert index layer %d down scale raw bytes=%d want %d", l, len(raw), numExperts*4)
			}
			downScale = make([]float32, numExperts)
			for i := 0; i < numExperts; i++ {
				downScale[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4 : i*4+4]))
			}
		}

		if guMat.Experts != numExperts || dnMat.Experts != numExperts {
			return nil, fmt.Errorf("GGUF expert index layer %d expert count gate_up=%d down=%d want %d", l, guMat.Experts, dnMat.Experts, numExperts)
		}
		if guMat.OutDim <= 0 || guMat.OutDim%2 != 0 || guMat.InDim <= 0 {
			return nil, fmt.Errorf("GGUF expert index layer %d invalid gate_up dims in=%d out=%d", l, guMat.InDim, guMat.OutDim)
		}
		intermediate := guMat.OutDim / 2
		if dnMat.InDim != intermediate || dnMat.OutDim != guMat.InDim {
			return nil, fmt.Errorf("GGUF expert index layer %d down dims [%d,%d] want [%d,%d]", l, dnMat.InDim, dnMat.OutDim, intermediate, guMat.InDim)
		}
		if downScale != nil && len(downScale) != numExperts {
			return nil, fmt.Errorf("GGUF expert index layer %d down scale len=%d want %d", l, len(downScale), numExperts)
		}
		if l == 0 {
			idx.Intermediate = intermediate
			idx.HiddenSize = guMat.InDim
			log.Printf("GGUF experts: gate_up [%d,%d,%d] %s, down [%d,%d,%d] %s",
				guMat.InDim, guMat.OutDim, guMat.Experts, guMat.QType,
				dnMat.InDim, dnMat.OutDim, dnMat.Experts, dnMat.QType)
		} else if idx.Intermediate != intermediate || idx.HiddenSize != guMat.InDim {
			return nil, fmt.Errorf("GGUF expert index layer %d shape hidden=%d/%d intermediate=%d/%d", l, guMat.InDim, idx.HiddenSize, intermediate, idx.Intermediate)
		}

		idx.entries[l] = ggufLayerExperts{
			gateUp:    guMat,
			down:      dnMat,
			downScale: downScale,
		}
	}

	log.Printf("GGUF expert index: %d layers × %d experts built in %s",
		numLayers, numExperts, time.Since(t0).Round(time.Millisecond))
	return idx, nil
}

// ggufWorkerScratch holds pre-allocated buffers for one expert worker.
type ggufWorkerScratch struct {
	wf32      []float32
	batchIn   []float32
	batchGU   []float32
	batchAct  []float32
	batchDown []float32
}

var ggufWorkerScratchPool = sync.Pool{New: func() any { return &ggufWorkerScratch{} }}

type ggufCPUPosWeight struct {
	pos int
	w   float32
}

type ggufCPUExpertGroupingScratch struct {
	expertUsers [][]ggufCPUPosWeight
	expertIDs   []int
	workerIDs   [][]int
	workerLoads []int
}

var ggufCPUExpertGroupingPool = sync.Pool{New: func() any { return &ggufCPUExpertGroupingScratch{} }}

type ggufCPUExpertBatchBuckets [10]uint64

type ggufCPUExpertTimingStats struct {
	Calls            uint64
	Positions        uint64
	WorkItems        uint64
	ActiveExperts    uint64
	Q4DirectRows     uint64
	Q4DequantRows    uint64
	Q8DirectRows     uint64
	Q8DequantRows    uint64
	Q4DirectBatches  ggufCPUExpertBatchBuckets
	Q4DequantBatches ggufCPUExpertBatchBuckets
	Q8DirectBatches  ggufCPUExpertBatchBuckets
	Q8DequantBatches ggufCPUExpertBatchBuckets
	NormNS           uint64
	CollectNS        uint64
	ScheduleNS       uint64
	GateNS           uint64
	ActNS            uint64
	DownNS           uint64
	ScatterNS        uint64
	PostNS           uint64
}

var ggufCPUExpertTimingCounters struct {
	calls            atomic.Uint64
	positions        atomic.Uint64
	workItems        atomic.Uint64
	activeExperts    atomic.Uint64
	q4DirectRows     atomic.Uint64
	q4DequantRows    atomic.Uint64
	q8DirectRows     atomic.Uint64
	q8DequantRows    atomic.Uint64
	q4DirectBatches  [10]atomic.Uint64
	q4DequantBatches [10]atomic.Uint64
	q8DirectBatches  [10]atomic.Uint64
	q8DequantBatches [10]atomic.Uint64
	normNS           atomic.Uint64
	collectNS        atomic.Uint64
	scheduleNS       atomic.Uint64
	gateNS           atomic.Uint64
	actNS            atomic.Uint64
	downNS           atomic.Uint64
	scatterNS        atomic.Uint64
	postNS           atomic.Uint64
}

func ggufCPUExpertBatchBucket(nPos int) int {
	switch {
	case nPos <= 1:
		return 0
	case nPos <= 3:
		return 1
	case nPos == 4:
		return 2
	case nPos == 5:
		return 3
	case nPos == 6:
		return 4
	case nPos == 7:
		return 5
	case nPos == 8:
		return 6
	case nPos <= 12:
		return 7
	case nPos <= 15:
		return 8
	default:
		return 9
	}
}

func ggufCPUExpertLoadBuckets(c *[10]atomic.Uint64) ggufCPUExpertBatchBuckets {
	var out ggufCPUExpertBatchBuckets
	for i := range out {
		out[i] = c[i].Load()
	}
	return out
}

func ggufCPUExpertStoreZeroBuckets(c *[10]atomic.Uint64) {
	for i := range c {
		c[i].Store(0)
	}
}

func ggufCPUExpertSubBuckets(a, b ggufCPUExpertBatchBuckets) ggufCPUExpertBatchBuckets {
	var out ggufCPUExpertBatchBuckets
	for i := range out {
		out[i] = a[i] - b[i]
	}
	return out
}

func ggufCPUExpertBatchBucketsString(b ggufCPUExpertBatchBuckets) string {
	return fmt.Sprintf("1:%d,2-3:%d,4:%d,5:%d,6:%d,7:%d,8:%d,9-12:%d,13-15:%d,16+:%d", b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9])
}

const diffusionGemmaDirectQuantMaxBatch = 8

func useDirectQ4GateUpRows(le ggufLayerExperts, nPos int) bool {
	return diffusionGemmaGGUFCPUQ4DirectEnabled() && simd.HasDotU4F32SIMD && nPos > 0 && nPos <= diffusionGemmaDirectQuantMaxBatch && le.gateUp != nil && le.gateUp.QType == gguf.QuantQ4_K
}

func useDirectQ8DownRows(le ggufLayerExperts, nPos int) bool {
	return diffusionGemmaGGUFCPUQ8DirectEnabled() && simd.HasDotI8F32SIMD && nPos > 0 && nPos <= diffusionGemmaDirectQuantMaxBatch && le.down != nil && le.down.QType == gguf.QuantQ8_0
}

func ggufCPUExpertTimingSnapshot() ggufCPUExpertTimingStats {
	return ggufCPUExpertTimingStats{
		Calls:            ggufCPUExpertTimingCounters.calls.Load(),
		Positions:        ggufCPUExpertTimingCounters.positions.Load(),
		WorkItems:        ggufCPUExpertTimingCounters.workItems.Load(),
		ActiveExperts:    ggufCPUExpertTimingCounters.activeExperts.Load(),
		Q4DirectRows:     ggufCPUExpertTimingCounters.q4DirectRows.Load(),
		Q4DequantRows:    ggufCPUExpertTimingCounters.q4DequantRows.Load(),
		Q8DirectRows:     ggufCPUExpertTimingCounters.q8DirectRows.Load(),
		Q8DequantRows:    ggufCPUExpertTimingCounters.q8DequantRows.Load(),
		Q4DirectBatches:  ggufCPUExpertLoadBuckets(&ggufCPUExpertTimingCounters.q4DirectBatches),
		Q4DequantBatches: ggufCPUExpertLoadBuckets(&ggufCPUExpertTimingCounters.q4DequantBatches),
		Q8DirectBatches:  ggufCPUExpertLoadBuckets(&ggufCPUExpertTimingCounters.q8DirectBatches),
		Q8DequantBatches: ggufCPUExpertLoadBuckets(&ggufCPUExpertTimingCounters.q8DequantBatches),
		NormNS:           ggufCPUExpertTimingCounters.normNS.Load(),
		CollectNS:        ggufCPUExpertTimingCounters.collectNS.Load(),
		ScheduleNS:       ggufCPUExpertTimingCounters.scheduleNS.Load(),
		GateNS:           ggufCPUExpertTimingCounters.gateNS.Load(),
		ActNS:            ggufCPUExpertTimingCounters.actNS.Load(),
		DownNS:           ggufCPUExpertTimingCounters.downNS.Load(),
		ScatterNS:        ggufCPUExpertTimingCounters.scatterNS.Load(),
		PostNS:           ggufCPUExpertTimingCounters.postNS.Load(),
	}
}

func ResetGGUFCPUExpertTimingStats() {
	ggufCPUExpertTimingCounters.calls.Store(0)
	ggufCPUExpertTimingCounters.positions.Store(0)
	ggufCPUExpertTimingCounters.workItems.Store(0)
	ggufCPUExpertTimingCounters.activeExperts.Store(0)
	ggufCPUExpertTimingCounters.q4DirectRows.Store(0)
	ggufCPUExpertTimingCounters.q4DequantRows.Store(0)
	ggufCPUExpertTimingCounters.q8DirectRows.Store(0)
	ggufCPUExpertTimingCounters.q8DequantRows.Store(0)
	ggufCPUExpertStoreZeroBuckets(&ggufCPUExpertTimingCounters.q4DirectBatches)
	ggufCPUExpertStoreZeroBuckets(&ggufCPUExpertTimingCounters.q4DequantBatches)
	ggufCPUExpertStoreZeroBuckets(&ggufCPUExpertTimingCounters.q8DirectBatches)
	ggufCPUExpertStoreZeroBuckets(&ggufCPUExpertTimingCounters.q8DequantBatches)
	ggufCPUExpertTimingCounters.normNS.Store(0)
	ggufCPUExpertTimingCounters.collectNS.Store(0)
	ggufCPUExpertTimingCounters.scheduleNS.Store(0)
	ggufCPUExpertTimingCounters.gateNS.Store(0)
	ggufCPUExpertTimingCounters.actNS.Store(0)
	ggufCPUExpertTimingCounters.downNS.Store(0)
	ggufCPUExpertTimingCounters.scatterNS.Store(0)
	ggufCPUExpertTimingCounters.postNS.Store(0)
}

func (s ggufCPUExpertTimingStats) Sub(base ggufCPUExpertTimingStats) ggufCPUExpertTimingStats {
	return ggufCPUExpertTimingStats{
		Calls:            s.Calls - base.Calls,
		Positions:        s.Positions - base.Positions,
		WorkItems:        s.WorkItems - base.WorkItems,
		ActiveExperts:    s.ActiveExperts - base.ActiveExperts,
		Q4DirectRows:     s.Q4DirectRows - base.Q4DirectRows,
		Q4DequantRows:    s.Q4DequantRows - base.Q4DequantRows,
		Q8DirectRows:     s.Q8DirectRows - base.Q8DirectRows,
		Q8DequantRows:    s.Q8DequantRows - base.Q8DequantRows,
		Q4DirectBatches:  ggufCPUExpertSubBuckets(s.Q4DirectBatches, base.Q4DirectBatches),
		Q4DequantBatches: ggufCPUExpertSubBuckets(s.Q4DequantBatches, base.Q4DequantBatches),
		Q8DirectBatches:  ggufCPUExpertSubBuckets(s.Q8DirectBatches, base.Q8DirectBatches),
		Q8DequantBatches: ggufCPUExpertSubBuckets(s.Q8DequantBatches, base.Q8DequantBatches),
		NormNS:           s.NormNS - base.NormNS,
		CollectNS:        s.CollectNS - base.CollectNS,
		ScheduleNS:       s.ScheduleNS - base.ScheduleNS,
		GateNS:           s.GateNS - base.GateNS,
		ActNS:            s.ActNS - base.ActNS,
		DownNS:           s.DownNS - base.DownNS,
		ScatterNS:        s.ScatterNS - base.ScatterNS,
		PostNS:           s.PostNS - base.PostNS,
	}
}

func diffusionGemmaGGUFCPUQ8DirectEnabled() bool {
	return diffusionGemmaGGUFCPUDirectPolicyEnabled("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_CPU_Q8_DIRECT")
}

func diffusionGemmaGGUFCPUQ4DirectEnabled() bool {
	return diffusionGemmaGGUFCPUDirectPolicyEnabled("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_CPU_Q4_DIRECT")
}

func diffusionGemmaGGUFCPUExpertLayerTraceEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_CPU_EXPERT_LAYER_TRACE")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func diffusionGemmaGGUFCPUDirectPolicyEnabled(name string) bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv(name)))
	if v == "" {
		// SIMD direct quantized row-dot is the default for small expert batches;
		// callers still gate on platform capability and batch size so larger batched
		// rows keep the faster dequant-once + Sdot reuse path.
		return true
	}
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func ggufQ4KExpertRowDot(m *gguf.ExpertMatrices, expert, row int, x []float32) (float32, error) {
	if m == nil || m.QType != gguf.QuantQ4_K || expert < 0 || expert >= m.Experts || row < 0 || row >= m.OutDim || len(x) < m.InDim || m.InDim%256 != 0 {
		return 0, fmt.Errorf("invalid Q4_K expert row dot expert=%d row=%d", expert, row)
	}
	blocks := m.InDim / 256
	rowBytes := blocks * 144
	rowIndex := expert*m.OutDim + row
	start := rowIndex * rowBytes
	end := start + rowBytes
	if start < 0 || end < start || end > len(m.Raw) {
		return 0, fmt.Errorf("Q4_K expert row raw outside expert=%d row=%d", expert, row)
	}
	return ggufQ4KRawRowDot(m.Raw[start:end], m.InDim, x), nil
}

func ggufQ4KRawRowDot(raw []byte, inDim int, x []float32) float32 {
	blocks := inDim / 256
	var sum float32
	for b := 0; b < blocks; b++ {
		blk := raw[b*144 : (b+1)*144]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[0:2]))
		dmin := half.F16ToF32(binary.LittleEndian.Uint16(blk[2:4]))
		sc := blk[4:16]
		qs := blk[16:144]
		var scales [8]float32
		var mins [8]float32
		for j := 0; j < 4; j++ {
			scales[j] = float32(sc[j]&63) * d
			mins[j] = float32(sc[j+4]&63) * dmin
		}
		for j := 4; j < 8; j++ {
			k := j - 4
			scales[j] = float32((sc[j+4]&0x0F)|((sc[k]>>6)<<4)) * d
			mins[j] = float32((sc[j+4]>>4)|((sc[k+4]>>6)<<4)) * dmin
		}
		xb := x[b*256 : b*256+256]
		for group := 0; group < 8; group++ {
			scale := scales[group]
			minv := mins[group]
			qoff := (group / 2) * 32
			qbytes := qs[qoff : qoff+32]
			xg := xb[group*32 : group*32+32]
			if group&1 == 0 {
				qdot, xsum, ok := simd.DotU4F32LowAndSum(qbytes, xg)
				if ok {
					sum += scale*qdot - minv*xsum
					continue
				}
			} else {
				qdot, xsum, ok := simd.DotU4F32HighAndSum(qbytes, xg)
				if ok {
					sum += scale*qdot - minv*xsum
					continue
				}
			}
			for i := 0; i < 32; i++ {
				qbyte := qbytes[i]
				qv := qbyte & 0x0F
				if group&1 != 0 {
					qv = qbyte >> 4
				}
				sum += (float32(qv)*scale - minv) * xg[i]
			}
		}
	}
	return sum
}

func ggufQ4KRawRowDot4(raw []byte, inDim int, x []float32, stride int) (float32, float32, float32, float32, bool) {
	if inDim%256 != 0 || stride < inDim || len(x) < 3*stride+inDim {
		return 0, 0, 0, 0, false
	}
	blocks := inDim / 256
	var s0, s1, s2, s3 float32
	for b := 0; b < blocks; b++ {
		blk := raw[b*144 : (b+1)*144]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[0:2]))
		dmin := half.F16ToF32(binary.LittleEndian.Uint16(blk[2:4]))
		sc := blk[4:16]
		qs := blk[16:144]
		var scales [8]float32
		var mins [8]float32
		for j := 0; j < 4; j++ {
			scales[j] = float32(sc[j]&63) * d
			mins[j] = float32(sc[j+4]&63) * dmin
		}
		for j := 4; j < 8; j++ {
			k := j - 4
			scales[j] = float32((sc[j+4]&0x0F)|((sc[k]>>6)<<4)) * d
			mins[j] = float32((sc[j+4]>>4)|((sc[k+4]>>6)<<4)) * dmin
		}
		xb := x[b*256:]
		for group := 0; group < 8; group++ {
			scale := scales[group]
			minv := mins[group]
			qoff := (group / 2) * 32
			qbytes := qs[qoff : qoff+32]
			xg := xb[group*32:]
			var d0, xs0, d1, xs1, d2, xs2, d3, xs3 float32
			var ok bool
			if group&1 == 0 {
				d0, xs0, d1, xs1, d2, xs2, d3, xs3, ok = simd.DotU4F32LowAndSumx4(qbytes, xg, stride)
			} else {
				d0, xs0, d1, xs1, d2, xs2, d3, xs3, ok = simd.DotU4F32HighAndSumx4(qbytes, xg, stride)
			}
			if ok {
				s0 += scale*d0 - minv*xs0
				s1 += scale*d1 - minv*xs1
				s2 += scale*d2 - minv*xs2
				s3 += scale*d3 - minv*xs3
				continue
			}
			for i := 0; i < 32; i++ {
				qbyte := qbytes[i]
				qv := qbyte & 0x0F
				if group&1 != 0 {
					qv = qbyte >> 4
				}
				coeff := float32(qv)*scale - minv
				off := b*256 + group*32 + i
				s0 += coeff * x[off]
				s1 += coeff * x[stride+off]
				s2 += coeff * x[2*stride+off]
				s3 += coeff * x[3*stride+off]
			}
		}
	}
	return s0, s1, s2, s3, true
}

func ggufQ4KExpertRowDotBatchTo(m *gguf.ExpertMatrices, expert, row int, x []float32, nPos int, dst []float32, dstStride int) error {
	if m == nil || m.QType != gguf.QuantQ4_K || expert < 0 || expert >= m.Experts || row < 0 || row >= m.OutDim || nPos <= 0 || len(x) < nPos*m.InDim || len(dst) < (nPos-1)*dstStride+1 || dstStride <= 0 || m.InDim%256 != 0 {
		return fmt.Errorf("invalid Q4_K expert row batch dot expert=%d row=%d nPos=%d", expert, row, nPos)
	}
	blocks := m.InDim / 256
	rowBytes := blocks * 144
	rowIndex := expert*m.OutDim + row
	start := rowIndex * rowBytes
	end := start + rowBytes
	if start < 0 || end < start || end > len(m.Raw) {
		return fmt.Errorf("Q4_K expert row batch raw outside expert=%d row=%d", expert, row)
	}
	raw := m.Raw[start:end]
	pos := 0
	for ; pos+4 <= nPos; pos += 4 {
		v0, v1, v2, v3, ok := ggufQ4KRawRowDot4(raw, m.InDim, x[pos*m.InDim:], m.InDim)
		if !ok {
			break
		}
		dst[pos*dstStride] = v0
		dst[(pos+1)*dstStride] = v1
		dst[(pos+2)*dstStride] = v2
		dst[(pos+3)*dstStride] = v3
	}
	for ; pos < nPos; pos++ {
		dst[pos*dstStride] = ggufQ4KRawRowDot(raw, m.InDim, x[pos*m.InDim:(pos+1)*m.InDim])
	}
	return nil
}

func ggufExpertSdotBatchTo(wRow, x []float32, nPos, inDim int, dst []float32, dstStride int) bool {
	if len(wRow) < inDim || nPos <= 0 || inDim <= 0 || len(x) < nPos*inDim || dstStride <= 0 || len(dst) < (nPos-1)*dstStride+1 {
		return false
	}
	pos := 0
	for ; pos+4 <= nPos; pos += 4 {
		d0, d1, d2, d3, ok := simd.Sdotx4(wRow[:inDim], x[pos*inDim:], inDim)
		if !ok {
			break
		}
		dst[pos*dstStride] = d0
		dst[(pos+1)*dstStride] = d1
		dst[(pos+2)*dstStride] = d2
		dst[(pos+3)*dstStride] = d3
	}
	for ; pos < nPos; pos++ {
		dst[pos*dstStride] = simd.Sdot(wRow[:inDim], x[pos*inDim:(pos+1)*inDim])
	}
	return true
}

func ggufQ8_0ExpertRowDot(m *gguf.ExpertMatrices, expert, row int, x []float32) (float32, error) {
	if m == nil || m.QType != gguf.QuantQ8_0 || expert < 0 || expert >= m.Experts || row < 0 || row >= m.OutDim || len(x) < m.InDim || m.InDim%32 != 0 {
		return 0, fmt.Errorf("invalid Q8_0 expert row dot expert=%d row=%d", expert, row)
	}
	blocks := m.InDim / 32
	rowBytes := blocks * 34
	rowIndex := expert*m.OutDim + row
	start := rowIndex * rowBytes
	end := start + rowBytes
	if start < 0 || end < start || end > len(m.Raw) {
		return 0, fmt.Errorf("Q8_0 expert row raw outside expert=%d row=%d", expert, row)
	}
	return ggufQ8_0RawRowDot(m.Raw[start:end], m.InDim, x), nil
}

func ggufQ8_0RawRowDot(raw []byte, inDim int, x []float32) float32 {
	blocks := inDim / 32
	var sum float32
	for b := 0; b < blocks; b++ {
		blk := raw[b*34 : (b+1)*34]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[:2]))
		qs := blk[2:34]
		xb := x[b*32 : b*32+32]
		blockSum, ok := simd.DotI8F32(qs, xb)
		if !ok {
			for i := 0; i < 32; i++ {
				blockSum += float32(int8(qs[i])) * xb[i]
			}
		}
		sum += d * blockSum
	}
	return sum
}

func ggufQ8_0RawRowDot4(raw []byte, inDim int, x []float32, stride int) (float32, float32, float32, float32, bool) {
	if inDim%32 != 0 || stride < inDim || len(x) < 3*stride+inDim {
		return 0, 0, 0, 0, false
	}
	blocks := inDim / 32
	var s0, s1, s2, s3 float32
	for b := 0; b < blocks; b++ {
		blk := raw[b*34 : (b+1)*34]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[:2]))
		qs := blk[2:34]
		v0, v1, v2, v3, ok := simd.DotI8F32x4(qs, x[b*32:], stride)
		if !ok {
			v0, v1, v2, v3 = 0, 0, 0, 0
			for i := 0; i < 32; i++ {
				qv := float32(int8(qs[i]))
				off := b*32 + i
				v0 += qv * x[off]
				v1 += qv * x[stride+off]
				v2 += qv * x[2*stride+off]
				v3 += qv * x[3*stride+off]
			}
		}
		s0 += d * v0
		s1 += d * v1
		s2 += d * v2
		s3 += d * v3
	}
	return s0, s1, s2, s3, true
}

func ggufQ8_0ExpertRowDotBatchTo(m *gguf.ExpertMatrices, expert, row int, x []float32, nPos int, dst []float32, dstStride int, scale float32) error {
	if m == nil || m.QType != gguf.QuantQ8_0 || expert < 0 || expert >= m.Experts || row < 0 || row >= m.OutDim || nPos <= 0 || len(x) < nPos*m.InDim || len(dst) < (nPos-1)*dstStride+1 || dstStride <= 0 || m.InDim%32 != 0 {
		return fmt.Errorf("invalid Q8_0 expert row batch dot expert=%d row=%d nPos=%d", expert, row, nPos)
	}
	blocks := m.InDim / 32
	rowBytes := blocks * 34
	rowIndex := expert*m.OutDim + row
	start := rowIndex * rowBytes
	end := start + rowBytes
	if start < 0 || end < start || end > len(m.Raw) {
		return fmt.Errorf("Q8_0 expert row batch raw outside expert=%d row=%d", expert, row)
	}
	raw := m.Raw[start:end]
	pos := 0
	for ; pos+4 <= nPos; pos += 4 {
		v0, v1, v2, v3, ok := ggufQ8_0RawRowDot4(raw, m.InDim, x[pos*m.InDim:], m.InDim)
		if !ok {
			break
		}
		dst[pos*dstStride] = scale * v0
		dst[(pos+1)*dstStride] = scale * v1
		dst[(pos+2)*dstStride] = scale * v2
		dst[(pos+3)*dstStride] = scale * v3
	}
	for ; pos < nPos; pos++ {
		dst[pos*dstStride] = scale * ggufQ8_0RawRowDot(raw, m.InDim, x[pos*m.InDim:(pos+1)*m.InDim])
	}
	return nil
}

func (s *ggufWorkerScratch) ensure(maxBatch, hiddenSize, intermediate int) error {
	wfNeed := hiddenSize
	inter2, okInter2 := checked.MulInt(intermediate, 2)
	if !okInter2 {
		return fmt.Errorf("GGUF expert scratch intermediate overflow %d", intermediate)
	}
	if inter2 > wfNeed {
		wfNeed = inter2
	}
	inNeed, okIn := checked.MulInt(maxBatch, hiddenSize)
	midNeed, okMid := checked.MulInt(maxBatch, intermediate)
	guNeed, okGU := checked.MulInt(maxBatch, inter2)
	if maxBatch <= 0 || hiddenSize <= 0 || intermediate <= 0 || !okIn || !okMid || !okGU {
		return fmt.Errorf("GGUF expert scratch size overflow batch=%d hidden=%d intermediate=%d", maxBatch, hiddenSize, intermediate)
	}
	grow := func(buf *[]float32, n int) {
		if cap(*buf) < n {
			*buf = make([]float32, n)
		} else {
			*buf = (*buf)[:n]
		}
	}
	grow(&s.wf32, wfNeed)
	grow(&s.batchIn, inNeed)
	grow(&s.batchGU, guNeed)
	grow(&s.batchAct, midNeed)
	grow(&s.batchDown, inNeed)
	return nil
}

// runGGUFCPUExpertsIndexed runs MoE experts using GGUF quantized weights.
// Uses dequant-once + SIMD Sdot for fast batched GEMV, matching the
// ggml_mul_mat_id strategy from llama.cpp.
func runGGUFCPUExpertsIndexed(op LayerOp, weights *TextWeights, scratch ForwardScratch, idx *GGUFExpertIndex) error {
	if weights == nil || idx == nil {
		return fmt.Errorf("GGUF CPU experts: missing weights or index")
	}
	fp := weights.ForwardPlan()
	if op.Layer < 0 || op.Layer >= len(fp.Layers) || op.Layer >= idx.NumLayers {
		return fmt.Errorf("layer %d outside plan/index", op.Layer)
	}
	lb := fp.Layers[op.Layer]
	preNorm2, err := loadFloatVector(weights, lb.PreFFNLayerNorm2)
	if err != nil {
		return err
	}
	hiddenSize := len(preNorm2)
	if hiddenSize <= 0 || len(scratch.Residual)%hiddenSize != 0 || len(scratch.MoeOut) < len(scratch.Residual) {
		return fmt.Errorf("GGUF CPU experts: invalid hidden/residual size hidden=%d residual=%d moe=%d", hiddenSize, len(scratch.Residual), len(scratch.MoeOut))
	}
	positions := len(scratch.Residual) / hiddenSize
	if positions <= 0 {
		return fmt.Errorf("GGUF CPU experts: no positions")
	}
	topK := scratch.TopKExperts
	if topK <= 0 {
		topK = len(scratch.TopKIDs) / positions
	}
	if topK <= 0 || len(scratch.TopKIDs) < positions*topK || len(scratch.TopKVals) < positions*topK {
		return fmt.Errorf("GGUF CPU experts: invalid top-k scratch positions=%d topK=%d ids=%d vals=%d", positions, topK, len(scratch.TopKIDs), len(scratch.TopKVals))
	}
	intermediate := idx.Intermediate
	if intermediate <= 0 || idx.HiddenSize != hiddenSize {
		return fmt.Errorf("GGUF CPU experts: index shape hidden=%d/%d intermediate=%d", idx.HiddenSize, hiddenSize, intermediate)
	}

	// Pre-norm all positions. Prefer the reusable ForwardScratch.Experts buffer
	// so CPU/SIMD GGUF expert fallback does not allocate a full
	// [positions,hidden] row block on every MoE call.
	normStart := time.Now()
	normedLen := positions * hiddenSize
	normedRows := scratch.Experts
	if len(normedRows) < normedLen {
		normedRows = make([]float32, normedLen)
	} else {
		normedRows = normedRows[:normedLen]
	}
	for pos := 0; pos < positions; pos++ {
		resRow := scratch.Residual[pos*hiddenSize : (pos+1)*hiddenSize]
		dst := normedRows[pos*hiddenSize : (pos+1)*hiddenSize]
		copy(dst, resRow)
		if !simd.RMSNormTo(dst, preNorm2, 1e-6) {
			return fmt.Errorf("pre_norm_2 rejected")
		}
	}
	ggufCPUExpertTimingCounters.normNS.Add(uint64(time.Since(normStart).Nanoseconds()))
	return runGGUFCPUExpertsIndexedWithNormedRows(op, weights, scratch, idx, normedRows)
}

func runGGUFCPUExpertsIndexedWithNormedRows(op LayerOp, weights *TextWeights, scratch ForwardScratch, idx *GGUFExpertIndex, normedRows []float32) error {
	if weights == nil || idx == nil {
		return fmt.Errorf("GGUF CPU experts: missing weights or index")
	}
	fp := weights.ForwardPlan()
	if op.Layer < 0 || op.Layer >= len(fp.Layers) || op.Layer >= idx.NumLayers {
		return fmt.Errorf("layer %d outside plan/index", op.Layer)
	}
	lb := fp.Layers[op.Layer]
	hiddenSize := idx.HiddenSize
	if hiddenSize <= 0 || len(normedRows)%hiddenSize != 0 || len(scratch.MoeOut) < len(normedRows) {
		return fmt.Errorf("GGUF CPU experts: invalid normed rows hidden=%d normed=%d moe=%d", hiddenSize, len(normedRows), len(scratch.MoeOut))
	}
	positions := len(normedRows) / hiddenSize
	if positions <= 0 {
		return fmt.Errorf("GGUF CPU experts: no positions")
	}
	topK := scratch.TopKExperts
	if topK <= 0 {
		topK = len(scratch.TopKIDs) / positions
	}
	if topK <= 0 || len(scratch.TopKIDs) < positions*topK || len(scratch.TopKVals) < positions*topK {
		return fmt.Errorf("GGUF CPU experts: invalid top-k scratch positions=%d topK=%d ids=%d vals=%d", positions, topK, len(scratch.TopKIDs), len(scratch.TopKVals))
	}
	intermediate := idx.Intermediate
	if intermediate <= 0 {
		return fmt.Errorf("GGUF CPU experts: invalid intermediate=%d", intermediate)
	}
	ggufCPUExpertTimingCounters.calls.Add(1)
	ggufCPUExpertTimingCounters.positions.Add(uint64(positions))
	for i := range scratch.MoeOut {
		scratch.MoeOut[i] = 0
	}

	groupScratch := ggufCPUExpertGroupingPool.Get().(*ggufCPUExpertGroupingScratch)
	defer func() {
		for i := range groupScratch.expertUsers {
			groupScratch.expertUsers[i] = groupScratch.expertUsers[i][:0]
		}
		groupScratch.expertIDs = groupScratch.expertIDs[:0]
		for i := range groupScratch.workerIDs {
			groupScratch.workerIDs[i] = groupScratch.workerIDs[i][:0]
		}
		for i := range groupScratch.workerLoads {
			groupScratch.workerLoads[i] = 0
		}
		ggufCPUExpertGroupingPool.Put(groupScratch)
	}()
	if cap(groupScratch.expertUsers) < idx.NumExperts {
		groupScratch.expertUsers = make([][]ggufCPUPosWeight, idx.NumExperts)
	}
	expertUsers := groupScratch.expertUsers[:idx.NumExperts]
	for i := range expertUsers {
		expertUsers[i] = expertUsers[i][:0]
	}

	// Collect experts and positions
	collectStart := time.Now()
	activeExperts := 0
	for pos := 0; pos < positions; pos++ {
		for k := 0; k < topK; k++ {
			eid := scratch.TopKIDs[pos*topK+k]
			if eid >= 0 && eid < idx.NumExperts {
				if len(expertUsers[eid]) == 0 {
					activeExperts++
				}
				expertUsers[eid] = append(expertUsers[eid], ggufCPUPosWeight{pos: pos, w: scratch.TopKVals[pos*topK+k]})
			}
		}
	}
	ggufCPUExpertTimingCounters.collectNS.Add(uint64(time.Since(collectStart).Nanoseconds()))
	ggufCPUExpertTimingCounters.activeExperts.Add(uint64(activeExperts))
	ggufCPUExpertTimingCounters.workItems.Add(uint64(positions * topK))

	le := idx.entries[op.Layer]

	scheduleStart := time.Now()
	numWorkers := runtime.NumCPU()
	if numWorkers > activeExperts {
		numWorkers = activeExperts
	}
	expertIDs := groupScratch.expertIDs[:0]
	for eid := range expertUsers {
		if len(expertUsers[eid]) == 0 {
			continue
		}
		expertIDs = append(expertIDs, eid)
	}
	groupScratch.expertIDs = expertIDs
	sort.Slice(expertIDs, func(i, j int) bool {
		li, lj := len(expertUsers[expertIDs[i]]), len(expertUsers[expertIDs[j]])
		if li == lj {
			return expertIDs[i] < expertIDs[j]
		}
		return li > lj
	})
	if cap(groupScratch.workerIDs) < numWorkers {
		groupScratch.workerIDs = make([][]int, numWorkers)
	}
	workerExpertIDs := groupScratch.workerIDs[:numWorkers]
	for i := range workerExpertIDs {
		workerExpertIDs[i] = workerExpertIDs[i][:0]
	}
	if cap(groupScratch.workerLoads) < numWorkers {
		groupScratch.workerLoads = make([]int, numWorkers)
	}
	workerLoads := groupScratch.workerLoads[:numWorkers]
	for i := range workerLoads {
		workerLoads[i] = 0
	}
	for _, eid := range expertIDs {
		best := 0
		for w := 1; w < numWorkers; w++ {
			if workerLoads[w] < workerLoads[best] {
				best = w
			}
		}
		workerExpertIDs[best] = append(workerExpertIDs[best], eid)
		workerLoads[best] += len(expertUsers[eid])
	}
	ggufCPUExpertTimingCounters.scheduleNS.Add(uint64(time.Since(scheduleStart).Nanoseconds()))

	var mergeMu sync.Mutex
	var wg sync.WaitGroup
	var errOnce sync.Once
	var firstErr error

	for w := 0; w < numWorkers; w++ {
		w := w
		idsForWorker := workerExpertIDs[w]
		if len(idsForWorker) == 0 {
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()

			maxBatch := 0
			for _, eid := range idsForWorker {
				if n := len(expertUsers[eid]); n > maxBatch {
					maxBatch = n
				}
			}

			ws := ggufWorkerScratchPool.Get().(*ggufWorkerScratch)
			defer ggufWorkerScratchPool.Put(ws)
			if err := ws.ensure(maxBatch, hiddenSize, intermediate); err != nil {
				errOnce.Do(func() { firstErr = err })
				return
			}

			for _, eid := range idsForWorker {
				users := expertUsers[eid]
				nPos := len(users)

				// Gather input rows
				batchIn := ws.batchIn[:nPos*hiddenSize]
				for i, u := range users {
					copy(batchIn[i*hiddenSize:(i+1)*hiddenSize], normedRows[u.pos*hiddenSize:(u.pos+1)*hiddenSize])
				}

				// Fused gate+up: dequant each row and dot with all batch inputs.
				// Output layout: [nPos, 1408] where [:704] is gate, [704:] is up.
				gateStart := time.Now()
				batchGU := ws.batchGU[:nPos*intermediate*2]
				outDimGU := intermediate * 2 // 1408
				// SIMD direct Q4_K row-dot uses a 4-output raw-row primitive and remains
				// the better full-path policy through nPos<=8; larger gate/up batches
				// favor dequant-once + Sdotx4 reuse end-to-end.
				useDirectQ4GateUp := useDirectQ4GateUpRows(le, nPos)
				for r := 0; r < outDimGU; r++ {
					if useDirectQ4GateUp {
						if err := ggufQ4KExpertRowDotBatchTo(le.gateUp, eid, r, batchIn, nPos, batchGU[r:], outDimGU); err != nil {
							errOnce.Do(func() { firstErr = fmt.Errorf("expert %d gate_up row %d direct Q4_K batch: %w", eid, r, err) })
							return
						}
						continue
					}
					if err := le.gateUp.DequantExpertRowTo(ws.wf32[:hiddenSize], eid, r); err != nil {
						errOnce.Do(func() { firstErr = fmt.Errorf("expert %d gate_up row %d: %w", eid, r, err) })
						return
					}
					wRow := ws.wf32[:hiddenSize]
					if !ggufExpertSdotBatchTo(wRow, batchIn, nPos, hiddenSize, batchGU[r:], outDimGU) {
						errOnce.Do(func() { firstErr = fmt.Errorf("expert %d gate_up row %d Sdot batch rejected", eid, r) })
						return
					}
				}
				if le.gateUp.QType == gguf.QuantQ4_K {
					bucket := ggufCPUExpertBatchBucket(nPos)
					if useDirectQ4GateUp {
						ggufCPUExpertTimingCounters.q4DirectRows.Add(uint64(outDimGU * nPos))
						ggufCPUExpertTimingCounters.q4DirectBatches[bucket].Add(1)
					} else {
						ggufCPUExpertTimingCounters.q4DequantRows.Add(uint64(outDimGU))
						ggufCPUExpertTimingCounters.q4DequantBatches[bucket].Add(1)
					}
				}
				ggufCPUExpertTimingCounters.gateNS.Add(uint64(time.Since(gateStart).Nanoseconds()))

				// Split gate and up, apply activation: GELU(gate) * up
				actStart := time.Now()
				batchAct := ws.batchAct[:nPos*intermediate]
				for b := 0; b < nPos; b++ {
					gateSlice := batchGU[b*outDimGU : b*outDimGU+intermediate]
					upSlice := batchGU[b*outDimGU+intermediate : (b+1)*outDimGU]
					actSlice := batchAct[b*intermediate : (b+1)*intermediate]
					if !simd.GELUExactMulTo(actSlice, gateSlice, upSlice) {
						errOnce.Do(func() { firstErr = fmt.Errorf("expert activation rejected") })
						return
					}
				}
				ggufCPUExpertTimingCounters.actNS.Add(uint64(time.Since(actStart).Nanoseconds()))

				// Down projection: dequant each row and dot with all batch act vectors
				downStart := time.Now()
				batchDown := ws.batchDown[:nPos*hiddenSize]
				dnOutDim := le.down.OutDim // 2816
				dnInDim := le.down.InDim   // 704
				if dnOutDim != hiddenSize || dnInDim != intermediate {
					errOnce.Do(func() {
						firstErr = fmt.Errorf("expert %d down shape [%d,%d] want [%d,%d]", eid, dnOutDim, dnInDim, hiddenSize, intermediate)
					})
					return
				}
				if le.downScale != nil && eid >= len(le.downScale) {
					errOnce.Do(func() { firstErr = fmt.Errorf("expert %d down scale outside len=%d", eid, len(le.downScale)) })
					return
				}
				// SIMD direct Q8_0 row-dot uses a 4-output raw-row primitive and remains
				// the better full-path policy through nPos<=8; larger expert-down batches
				// favor dequant-once + Sdotx4 reuse end-to-end.
				useDirectQ8Down := useDirectQ8DownRows(le, nPos)
				for r := 0; r < dnOutDim; r++ {
					if useDirectQ8Down {
						scale := float32(1)
						if le.downScale != nil {
							scale = le.downScale[eid]
						}
						if err := ggufQ8_0ExpertRowDotBatchTo(le.down, eid, r, batchAct, nPos, batchDown[r:], hiddenSize, scale); err != nil {
							errOnce.Do(func() { firstErr = fmt.Errorf("expert %d down row %d direct Q8 batch: %w", eid, r, err) })
							return
						}
						continue
					}
					if err := le.down.DequantExpertRowTo(ws.wf32[:dnInDim], eid, r); err != nil {
						errOnce.Do(func() { firstErr = fmt.Errorf("expert %d down row %d: %w", eid, r, err) })
						return
					}
					if le.downScale != nil {
						s := le.downScale[eid]
						for j := 0; j < dnInDim; j++ {
							ws.wf32[j] *= s
						}
					}
					wRow := ws.wf32[:dnInDim]
					if !ggufExpertSdotBatchTo(wRow, batchAct, nPos, dnInDim, batchDown[r:], hiddenSize) {
						errOnce.Do(func() { firstErr = fmt.Errorf("expert %d down row %d Sdot batch rejected", eid, r) })
						return
					}
				}
				if le.down.QType == gguf.QuantQ8_0 {
					bucket := ggufCPUExpertBatchBucket(nPos)
					if useDirectQ8Down {
						ggufCPUExpertTimingCounters.q8DirectRows.Add(uint64(dnOutDim * nPos))
						ggufCPUExpertTimingCounters.q8DirectBatches[bucket].Add(1)
					} else {
						ggufCPUExpertTimingCounters.q8DequantRows.Add(uint64(dnOutDim))
						ggufCPUExpertTimingCounters.q8DequantBatches[bucket].Add(1)
					}
				}
				ggufCPUExpertTimingCounters.downNS.Add(uint64(time.Since(downStart).Nanoseconds()))

				// Scatter weighted outputs directly into the shared MoE buffer. This avoids
				// allocating one full [positions,hidden] output buffer per worker.
				scatterStart := time.Now()
				mergeMu.Lock()
				for i, u := range users {
					expertOut := batchDown[i*hiddenSize : (i+1)*hiddenSize]
					dst := scratch.MoeOut[u.pos*hiddenSize : (u.pos+1)*hiddenSize]
					simd.Saxpy(u.w, expertOut, dst)
				}
				mergeMu.Unlock()
				ggufCPUExpertTimingCounters.scatterNS.Add(uint64(time.Since(scatterStart).Nanoseconds()))
			}
		}()
	}
	wg.Wait()
	if firstErr != nil {
		return firstErr
	}

	postStart := time.Now()
	postNorm2, err := loadFloatVector(weights, lb.PostFFNLayerNorm2)
	if err != nil {
		return err
	}
	for off := 0; off < len(scratch.MoeOut); off += hiddenSize {
		if !simd.RMSNormTo(scratch.MoeOut[off:off+hiddenSize], postNorm2, 1e-6) {
			return fmt.Errorf("post_norm_2 rejected")
		}
	}
	ggufCPUExpertTimingCounters.postNS.Add(uint64(time.Since(postStart).Nanoseconds()))
	return nil
}

func runGGUFCPUExpertsGroupedNoPostNorm(op LayerOp, scratch ForwardScratch, idx *GGUFExpertIndex, normedRows []float32, groupedArrays SelectedExpertGroupedArrays) error {
	if idx == nil {
		return fmt.Errorf("GGUF CPU grouped experts: missing index")
	}
	if op.Layer < 0 || op.Layer >= idx.NumLayers {
		return fmt.Errorf("layer %d outside expert index", op.Layer)
	}
	if err := groupedArrays.Validate(); err != nil {
		return err
	}
	hiddenSize := idx.HiddenSize
	if hiddenSize <= 0 || len(normedRows)%hiddenSize != 0 || len(scratch.MoeOut) < len(normedRows) {
		return fmt.Errorf("GGUF CPU grouped experts: invalid normed rows hidden=%d normed=%d moe=%d", hiddenSize, len(normedRows), len(scratch.MoeOut))
	}
	positions := len(normedRows) / hiddenSize
	if positions <= 0 {
		return fmt.Errorf("GGUF CPU grouped experts: no positions")
	}
	intermediate := idx.Intermediate
	if intermediate <= 0 {
		return fmt.Errorf("GGUF CPU grouped experts: invalid intermediate=%d", intermediate)
	}
	if len(groupedArrays.ActiveExperts) == 0 || len(groupedArrays.WorkPositions) == 0 {
		return nil
	}
	for _, pos := range groupedArrays.WorkPositions {
		if pos < 0 || pos >= positions {
			return fmt.Errorf("GGUF CPU grouped experts: work position %d outside [0,%d)", pos, positions)
		}
	}
	ggufCPUExpertTimingCounters.calls.Add(1)
	ggufCPUExpertTimingCounters.positions.Add(uint64(positions))
	ggufCPUExpertTimingCounters.activeExperts.Add(uint64(len(groupedArrays.ActiveExperts)))
	ggufCPUExpertTimingCounters.workItems.Add(uint64(len(groupedArrays.WorkPositions)))

	le := idx.entries[op.Layer]
	type groupRef struct {
		group       int
		expert      int
		start, end  int
		workItemCnt int
	}
	groups := make([]groupRef, 0, len(groupedArrays.ActiveExperts))
	for groupIdx, eid := range groupedArrays.ActiveExperts {
		if eid < 0 || eid >= idx.NumExperts {
			return fmt.Errorf("GGUF CPU grouped experts: active expert %d outside [0,%d)", eid, idx.NumExperts)
		}
		start, end := groupedArrays.Offsets[groupIdx], groupedArrays.Offsets[groupIdx+1]
		if end <= start {
			continue
		}
		groups = append(groups, groupRef{group: groupIdx, expert: eid, start: start, end: end, workItemCnt: end - start})
	}
	if len(groups) == 0 {
		return nil
	}
	scheduleStart := time.Now()
	sort.Slice(groups, func(i, j int) bool {
		if groups[i].workItemCnt == groups[j].workItemCnt {
			return groups[i].expert < groups[j].expert
		}
		return groups[i].workItemCnt > groups[j].workItemCnt
	})
	numWorkers := runtime.NumCPU()
	if numWorkers > len(groups) {
		numWorkers = len(groups)
	}
	workerGroups := make([][]int, numWorkers)
	workerLoads := make([]int, numWorkers)
	for gi, group := range groups {
		best := 0
		for w := 1; w < numWorkers; w++ {
			if workerLoads[w] < workerLoads[best] {
				best = w
			}
		}
		workerGroups[best] = append(workerGroups[best], gi)
		workerLoads[best] += group.workItemCnt
	}
	ggufCPUExpertTimingCounters.scheduleNS.Add(uint64(time.Since(scheduleStart).Nanoseconds()))

	var mergeMu sync.Mutex
	var wg sync.WaitGroup
	var errOnce sync.Once
	var firstErr error
	for w := 0; w < numWorkers; w++ {
		idsForWorker := workerGroups[w]
		if len(idsForWorker) == 0 {
			continue
		}
		wg.Add(1)
		go func(groupIndexes []int) {
			defer wg.Done()
			maxBatch := 0
			for _, gi := range groupIndexes {
				if n := groups[gi].workItemCnt; n > maxBatch {
					maxBatch = n
				}
			}
			ws := ggufWorkerScratchPool.Get().(*ggufWorkerScratch)
			defer ggufWorkerScratchPool.Put(ws)
			if err := ws.ensure(maxBatch, hiddenSize, intermediate); err != nil {
				errOnce.Do(func() { firstErr = err })
				return
			}
			for _, gi := range groupIndexes {
				group := groups[gi]
				eid := group.expert
				nPos := group.workItemCnt
				batchIn := ws.batchIn[:nPos*hiddenSize]
				for i := 0; i < nPos; i++ {
					pos := groupedArrays.WorkPositions[group.start+i]
					copy(batchIn[i*hiddenSize:(i+1)*hiddenSize], normedRows[pos*hiddenSize:(pos+1)*hiddenSize])
				}

				gateStart := time.Now()
				batchGU := ws.batchGU[:nPos*intermediate*2]
				outDimGU := intermediate * 2
				useDirectQ4GateUp := useDirectQ4GateUpRows(le, nPos)
				for r := 0; r < outDimGU; r++ {
					if useDirectQ4GateUp {
						if err := ggufQ4KExpertRowDotBatchTo(le.gateUp, eid, r, batchIn, nPos, batchGU[r:], outDimGU); err != nil {
							errOnce.Do(func() { firstErr = fmt.Errorf("expert %d gate_up row %d direct Q4_K batch: %w", eid, r, err) })
							return
						}
						continue
					}
					if err := le.gateUp.DequantExpertRowTo(ws.wf32[:hiddenSize], eid, r); err != nil {
						errOnce.Do(func() { firstErr = fmt.Errorf("expert %d gate_up row %d: %w", eid, r, err) })
						return
					}
					if !ggufExpertSdotBatchTo(ws.wf32[:hiddenSize], batchIn, nPos, hiddenSize, batchGU[r:], outDimGU) {
						errOnce.Do(func() { firstErr = fmt.Errorf("expert %d gate_up row %d Sdot batch rejected", eid, r) })
						return
					}
				}
				if le.gateUp.QType == gguf.QuantQ4_K {
					bucket := ggufCPUExpertBatchBucket(nPos)
					if useDirectQ4GateUp {
						ggufCPUExpertTimingCounters.q4DirectRows.Add(uint64(outDimGU * nPos))
						ggufCPUExpertTimingCounters.q4DirectBatches[bucket].Add(1)
					} else {
						ggufCPUExpertTimingCounters.q4DequantRows.Add(uint64(outDimGU))
						ggufCPUExpertTimingCounters.q4DequantBatches[bucket].Add(1)
					}
				}
				ggufCPUExpertTimingCounters.gateNS.Add(uint64(time.Since(gateStart).Nanoseconds()))

				actStart := time.Now()
				batchAct := ws.batchAct[:nPos*intermediate]
				for b := 0; b < nPos; b++ {
					gateSlice := batchGU[b*outDimGU : b*outDimGU+intermediate]
					upSlice := batchGU[b*outDimGU+intermediate : (b+1)*outDimGU]
					actSlice := batchAct[b*intermediate : (b+1)*intermediate]
					if !simd.GELUExactMulTo(actSlice, gateSlice, upSlice) {
						errOnce.Do(func() { firstErr = fmt.Errorf("expert activation rejected") })
						return
					}
				}
				ggufCPUExpertTimingCounters.actNS.Add(uint64(time.Since(actStart).Nanoseconds()))

				downStart := time.Now()
				batchDown := ws.batchDown[:nPos*hiddenSize]
				dnOutDim := le.down.OutDim
				dnInDim := le.down.InDim
				if dnOutDim != hiddenSize || dnInDim != intermediate {
					errOnce.Do(func() {
						firstErr = fmt.Errorf("expert %d down shape [%d,%d] want [%d,%d]", eid, dnOutDim, dnInDim, hiddenSize, intermediate)
					})
					return
				}
				if le.downScale != nil && eid >= len(le.downScale) {
					errOnce.Do(func() { firstErr = fmt.Errorf("expert %d down scale outside len=%d", eid, len(le.downScale)) })
					return
				}
				useDirectQ8Down := useDirectQ8DownRows(le, nPos)
				for r := 0; r < dnOutDim; r++ {
					if useDirectQ8Down {
						scale := float32(1)
						if le.downScale != nil {
							scale = le.downScale[eid]
						}
						if err := ggufQ8_0ExpertRowDotBatchTo(le.down, eid, r, batchAct, nPos, batchDown[r:], hiddenSize, scale); err != nil {
							errOnce.Do(func() { firstErr = fmt.Errorf("expert %d down row %d direct Q8 batch: %w", eid, r, err) })
							return
						}
						continue
					}
					if err := le.down.DequantExpertRowTo(ws.wf32[:dnInDim], eid, r); err != nil {
						errOnce.Do(func() { firstErr = fmt.Errorf("expert %d down row %d: %w", eid, r, err) })
						return
					}
					if le.downScale != nil {
						s := le.downScale[eid]
						for j := 0; j < dnInDim; j++ {
							ws.wf32[j] *= s
						}
					}
					if !ggufExpertSdotBatchTo(ws.wf32[:dnInDim], batchAct, nPos, dnInDim, batchDown[r:], hiddenSize) {
						errOnce.Do(func() { firstErr = fmt.Errorf("expert %d down row %d Sdot batch rejected", eid, r) })
						return
					}
				}
				if le.down.QType == gguf.QuantQ8_0 {
					bucket := ggufCPUExpertBatchBucket(nPos)
					if useDirectQ8Down {
						ggufCPUExpertTimingCounters.q8DirectRows.Add(uint64(dnOutDim * nPos))
						ggufCPUExpertTimingCounters.q8DirectBatches[bucket].Add(1)
					} else {
						ggufCPUExpertTimingCounters.q8DequantRows.Add(uint64(dnOutDim))
						ggufCPUExpertTimingCounters.q8DequantBatches[bucket].Add(1)
					}
				}
				ggufCPUExpertTimingCounters.downNS.Add(uint64(time.Since(downStart).Nanoseconds()))

				scatterStart := time.Now()
				mergeMu.Lock()
				for i := 0; i < nPos; i++ {
					workIdx := group.start + i
					pos := groupedArrays.WorkPositions[workIdx]
					weight := groupedArrays.WorkWeights[workIdx]
					expertOut := batchDown[i*hiddenSize : (i+1)*hiddenSize]
					dst := scratch.MoeOut[pos*hiddenSize : (pos+1)*hiddenSize]
					simd.Saxpy(weight, expertOut, dst)
				}
				mergeMu.Unlock()
				ggufCPUExpertTimingCounters.scatterNS.Add(uint64(time.Since(scatterStart).Nanoseconds()))
			}
		}(idsForWorker)
	}
	wg.Wait()
	if firstErr != nil {
		return firstErr
	}
	return nil
}

// RunGGUFExpertMLP runs a single expert MLP using GGUF quantized weights.
// Input normedRow[hiddenSize], output expertOut[hiddenSize].
// Dequants gate_up and down on the fly using Sdot.
func (idx *GGUFExpertIndex) RunGGUFExpertMLP(layer, expertID int, normedRow, expertOut []float32) error {
	if idx == nil || layer < 0 || layer >= idx.NumLayers || expertID < 0 || expertID >= idx.NumExperts {
		return fmt.Errorf("GGUF expert MLP: invalid layer %d expert %d", layer, expertID)
	}
	le := idx.entries[layer]
	hiddenSize := idx.HiddenSize
	intermediate := idx.Intermediate
	if hiddenSize <= 0 || intermediate <= 0 || len(normedRow) < hiddenSize || len(expertOut) < hiddenSize || le.gateUp.InDim != hiddenSize || le.gateUp.OutDim != intermediate*2 || le.down.InDim != intermediate || le.down.OutDim != hiddenSize {
		return fmt.Errorf("GGUF expert MLP: shape mismatch hidden=%d intermediate=%d normed=%d out=%d gate_up=[%d,%d] down=[%d,%d]", hiddenSize, intermediate, len(normedRow), len(expertOut), le.gateUp.OutDim, le.gateUp.InDim, le.down.OutDim, le.down.InDim)
	}
	outDimGU := intermediate * 2

	// Dequant row buffer
	wf32 := make([]float32, hiddenSize)

	// Gate+up projection: dequant each output row, dot with input
	guOut := make([]float32, outDimGU)
	useDirectQ4GateUp := useDirectQ4GateUpRows(le, 1)
	for r := 0; r < outDimGU; r++ {
		if useDirectQ4GateUp {
			v, err := ggufQ4KExpertRowDot(le.gateUp, expertID, r, normedRow)
			if err != nil {
				return err
			}
			guOut[r] = v
			continue
		}
		if err := le.gateUp.DequantExpertRowTo(wf32[:hiddenSize], expertID, r); err != nil {
			return fmt.Errorf("expert %d gate_up row %d: %w", expertID, r, err)
		}
		guOut[r] = simd.Sdot(wf32[:hiddenSize], normedRow)
	}
	if le.gateUp.QType == gguf.QuantQ4_K {
		if useDirectQ4GateUp {
			ggufCPUExpertTimingCounters.q4DirectRows.Add(uint64(outDimGU))
			ggufCPUExpertTimingCounters.q4DirectBatches[0].Add(1)
		} else {
			ggufCPUExpertTimingCounters.q4DequantRows.Add(uint64(outDimGU))
			ggufCPUExpertTimingCounters.q4DequantBatches[0].Add(1)
		}
	}

	// Split gate and up, apply GELU(gate) * up
	actOut := make([]float32, intermediate)
	if !simd.GELUExactMulTo(actOut, guOut[:intermediate], guOut[intermediate:]) {
		return fmt.Errorf("GGUF expert MLP: activation rejected")
	}

	// Down projection: dequant each output row, dot with activation
	downBuf := make([]float32, intermediate)
	useDirectQ8Down := useDirectQ8DownRows(le, 1)
	for r := 0; r < hiddenSize; r++ {
		if useDirectQ8Down {
			v, err := ggufQ8_0ExpertRowDot(le.down, expertID, r, actOut)
			if err != nil {
				return err
			}
			expertOut[r] = v
			continue
		}
		if err := le.down.DequantExpertRowTo(downBuf[:intermediate], expertID, r); err != nil {
			return fmt.Errorf("expert %d down row %d: %w", expertID, r, err)
		}
		expertOut[r] = simd.Sdot(downBuf[:intermediate], actOut)
	}
	if le.down.QType == gguf.QuantQ8_0 {
		if useDirectQ8Down {
			ggufCPUExpertTimingCounters.q8DirectRows.Add(uint64(hiddenSize))
			ggufCPUExpertTimingCounters.q8DirectBatches[0].Add(1)
		} else {
			ggufCPUExpertTimingCounters.q8DequantRows.Add(uint64(hiddenSize))
			ggufCPUExpertTimingCounters.q8DequantBatches[0].Add(1)
		}
	}

	// Apply per-expert scale if present
	if le.downScale != nil {
		if expertID >= len(le.downScale) {
			return fmt.Errorf("GGUF expert MLP: expert %d down scale outside len=%d", expertID, len(le.downScale))
		}
		scale := le.downScale[expertID]
		for i := 0; i < hiddenSize; i++ {
			expertOut[i] *= scale
		}
	}

	return nil
}
