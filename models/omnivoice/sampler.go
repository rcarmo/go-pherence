package omnivoice

import (
	"fmt"
	"math"
	"strconv"
)

const omnivoiceTopKRatioDefault = 0.1

// GumbelNoise carries caller-supplied randomness for Gumbel perturbation.
//
// Exactly one source may be provided:
//   - Uniforms: values in [0, 1], converted with OmniVoice's upstream formula
//     -log(-log(u + 1e-10) + 1e-10)
//   - Gumbels: precomputed Gumbel noise values to add directly
//
// No method in this file draws randomness internally. This makes parity tests
// deterministic without claiming Go's RNG matches torch.
type GumbelNoise struct {
	Uniforms []float32
	Gumbels  []float32
}

// SamplerWorkspace owns reusable scratch for OmniVoice's iterative sampling
// primitives. It is not safe for concurrent use and never grows implicitly.
type SamplerWorkspace struct {
	codebooks     int
	maxTargetLen  int
	vocabSize     int
	maxPositions  int
	vocabScratch  []float32
	vocabScratch2 []float32
	topVals       []float32
	topIdx        []int
	posScratch    []float32
}

// NewSamplerWorkspace reserves scratch once for class- and position-sampling
// primitives over one batch item. maxTargetLen is the per-item audio-token
// length; rows are flattened in [codebook][time] order.
func NewSamplerWorkspace(numCodebooks, maxTargetLen, vocabSize int) (*SamplerWorkspace, error) {
	if numCodebooks <= 0 {
		return nil, fmt.Errorf("omnivoice: numCodebooks must be > 0")
	}
	if maxTargetLen <= 0 {
		return nil, fmt.Errorf("omnivoice: maxTargetLen must be > 0")
	}
	if vocabSize <= 0 {
		return nil, fmt.Errorf("omnivoice: vocabSize must be > 0")
	}
	rows, ok := product(numCodebooks, maxTargetLen)
	if !ok {
		return nil, fmt.Errorf("omnivoice: sampler shape overflow")
	}
	maxSpan := vocabSize
	if rows > maxSpan {
		maxSpan = rows
	}
	return &SamplerWorkspace{
		codebooks:     numCodebooks,
		maxTargetLen:  maxTargetLen,
		vocabSize:     vocabSize,
		maxPositions:  rows,
		vocabScratch:  make([]float32, vocabSize),
		vocabScratch2: make([]float32, vocabSize),
		topVals:       make([]float32, maxSpan),
		topIdx:        make([]int, maxSpan),
		posScratch:    make([]float32, rows),
	}, nil
}

// ScratchBytes returns float/int backing storage, excluding Go slice headers.
func (w *SamplerWorkspace) ScratchBytes() int {
	if w == nil {
		return 0
	}
	return 4*(len(w.vocabScratch)+len(w.vocabScratch2)+len(w.topVals)+len(w.posScratch)) + len(w.topIdx)*strconv.IntSize/8
}

// TimeStepsInto computes OmniVoice's shifted linspace schedule into dst. dst
// must have length numStep+1. The transform matches upstream:
// t_shift * t / (1 + (t_shift - 1) * t)
func TimeStepsInto(dst []float32, tStart, tEnd float32, numStep int, tShift float32) error {
	if numStep <= 0 {
		return fmt.Errorf("omnivoice: numStep must be > 0")
	}
	if len(dst) != numStep+1 {
		return fmt.Errorf("omnivoice: timestep length mismatch")
	}
	for i := 0; i <= numStep; i++ {
		t := tEnd
		if i != numStep {
			t = tStart + (tEnd-tStart)*(float32(i)/float32(numStep))
		}
		dst[i] = tShift * t / (1 + (tShift-1)*t)
	}
	return nil
}

// BuildUnmaskScheduleInto computes the per-step number of positions to reveal
// for one target length, matching OmniVoice's ceil(diff * total_mask) rule with
// the final step forced to consume the remaining masked positions.
func BuildUnmaskScheduleInto(dst []int, targetLen, numCodebooks int, timesteps []float32) error {
	if targetLen <= 0 {
		return fmt.Errorf("omnivoice: targetLen must be > 0")
	}
	if numCodebooks <= 0 {
		return fmt.Errorf("omnivoice: numCodebooks must be > 0")
	}
	if len(timesteps) < 2 || len(dst) != len(timesteps)-1 {
		return fmt.Errorf("omnivoice: schedule length mismatch")
	}
	totalMask, ok := product(targetLen, numCodebooks)
	if !ok {
		return fmt.Errorf("omnivoice: schedule size overflow")
	}
	rem := totalMask
	for step := range dst {
		var n int
		if step == len(dst)-1 {
			n = rem
		} else {
			delta := float64(timesteps[step+1] - timesteps[step])
			if delta < 0 || math.IsNaN(delta) || math.IsInf(delta, 0) {
				return fmt.Errorf("omnivoice: invalid timestep delta")
			}
			n = int(math.Ceil(float64(totalMask) * delta))
			if n > rem {
				n = rem
			}
		}
		dst[step] = n
		rem -= n
	}
	return nil
}

func (w *SamplerWorkspace) rowsFor(targetLen int) (int, error) {
	if w == nil {
		return 0, fmt.Errorf("omnivoice: nil sampler workspace")
	}
	if targetLen <= 0 || targetLen > w.maxTargetLen {
		return 0, fmt.Errorf("omnivoice: targetLen=%d outside [1,%d]", targetLen, w.maxTargetLen)
	}
	rows, ok := product(w.codebooks, targetLen)
	if !ok {
		return 0, fmt.Errorf("omnivoice: sampler shape overflow")
	}
	return rows, nil
}

func (w *SamplerWorkspace) validateLogitShape(buf []float32, rows int, what string) error {
	n, ok := product(rows, w.vocabSize)
	if !ok || len(buf) != n {
		return fmt.Errorf("omnivoice: %s shape mismatch", what)
	}
	return nil
}

// GuidedLogProbsInto computes the classifier-free-guided log-probabilities over
// flattened [codebook][time][vocab] logits. condLogits and uncondLogits must be
// row-major with identical shapes when guidanceScale != 0.
func (w *SamplerWorkspace) GuidedLogProbsInto(dst, condLogits, uncondLogits []float32, targetLen int, guidanceScale float32, audioMaskID int) error {
	rows, err := w.rowsFor(targetLen)
	if err != nil {
		return err
	}
	if audioMaskID < 0 || audioMaskID >= w.vocabSize {
		return fmt.Errorf("omnivoice: audioMaskID=%d outside [0,%d)", audioMaskID, w.vocabSize)
	}
	if err = w.validateLogitShape(dst, rows, "dst"); err != nil {
		return err
	}
	if err = w.validateLogitShape(condLogits, rows, "condLogits"); err != nil {
		return err
	}
	if guidanceScale != 0 {
		if err = w.validateLogitShape(uncondLogits, rows, "uncondLogits"); err != nil {
			return err
		}
	}
	for row := 0; row < rows; row++ {
		off := row * w.vocabSize
		dstRow := dst[off : off+w.vocabSize]
		copy(dstRow, condLogits[off:off+w.vocabSize])
		logSoftmaxSIMDInPlace(dstRow, w.vocabScratch2)
		if guidanceScale != 0 {
			copy(w.vocabScratch, uncondLogits[off:off+w.vocabSize])
			logSoftmaxSIMDInPlace(w.vocabScratch, w.vocabScratch2)
			for i := range dstRow {
				dstRow[i] = dstRow[i] + guidanceScale*(dstRow[i]-w.vocabScratch[i])
			}
			logSoftmaxSIMDInPlace(dstRow, w.vocabScratch2)
		}
		dstRow[audioMaskID] = float32(math.Inf(-1))
	}
	return nil
}

// FilterTopKInto keeps ceil(ratio * vocab) entries per row and fills the rest
// with -Inf, matching OmniVoice's _filter_top_k helper.
func (w *SamplerWorkspace) FilterTopKInto(dst, logProbs []float32, targetLen int, ratio float64) error {
	rows, err := w.rowsFor(targetLen)
	if err != nil {
		return err
	}
	if err = w.validateLogitShape(dst, rows, "dst"); err != nil {
		return err
	}
	if err = w.validateLogitShape(logProbs, rows, "logProbs"); err != nil {
		return err
	}
	if !(ratio > 0) || math.IsNaN(float64(ratio)) || math.IsInf(float64(ratio), 0) {
		return fmt.Errorf("omnivoice: ratio must be finite and > 0")
	}
	k := int(math.Ceil(float64(ratio) * float64(w.vocabSize)))
	if k < 1 {
		k = 1
	}
	if k > w.vocabSize {
		k = w.vocabSize
	}
	for row := 0; row < rows; row++ {
		off := row * w.vocabSize
		out := dst[off : off+w.vocabSize]
		count := selectTopKStable(w.topVals, w.topIdx, logProbs[off:off+w.vocabSize], k)
		for i := range out {
			out[i] = float32(math.Inf(-1))
		}
		for i := 0; i < count; i++ {
			out[w.topIdx[i]] = w.topVals[i]
		}
	}
	return nil
}

// GumbelPerturbInto applies OmniVoice's temperature/Gumbel transform:
// logits/temperature + gumbel. If temperature <= 0, logits are copied exactly
// and noise is ignored.
func (w *SamplerWorkspace) GumbelPerturbInto(dst, logits []float32, temperature float32, noise GumbelNoise) error {
	if len(dst) != len(logits) {
		return fmt.Errorf("omnivoice: gumbel dst/logit length mismatch")
	}
	if temperature <= 0 {
		copy(dst, logits)
		return nil
	}
	if err := validateNoise(noise, len(logits)); err != nil {
		return err
	}
	inv := float32(1) / temperature
	if len(noise.Gumbels) != 0 {
		for i, v := range logits {
			dst[i] = v*inv + noise.Gumbels[i]
		}
		return nil
	}
	for i, v := range logits {
		u := float64(noise.Uniforms[i])
		g := -math.Log(-math.Log(u+1e-10) + 1e-10)
		dst[i] = v*inv + float32(g)
	}
	return nil
}

// PredictTokensWithConfidenceInto matches OmniVoice's token prediction logic
// once guided log-probabilities are already available. predTokens and
// confidence are flattened in [codebook][time] order.
func (w *SamplerWorkspace) PredictTokensWithConfidenceInto(predTokens []int, confidence []float32, logProbs []float32, targetLen int, classTemperature float32, topKRatio float64, noise GumbelNoise) error {
	rows, err := w.rowsFor(targetLen)
	if err != nil {
		return err
	}
	if len(predTokens) != rows || len(confidence) != rows {
		return fmt.Errorf("omnivoice: prediction output shape mismatch")
	}
	if err = w.validateLogitShape(logProbs, rows, "logProbs"); err != nil {
		return err
	}
	if classTemperature > 0 {
		if err = validateNoise(noise, rows*w.vocabSize); err != nil {
			return err
		}
	}
	for row := 0; row < rows; row++ {
		off := row * w.vocabSize
		lp := logProbs[off : off+w.vocabSize]
		bestIdx, bestVal := argmaxStable(lp)
		predTokens[row], confidence[row] = bestIdx, bestVal
	}
	if classTemperature <= 0 {
		return nil
	}
	k := int(math.Ceil(float64(topKRatio) * float64(w.vocabSize)))
	if !(topKRatio > 0) || math.IsNaN(float64(topKRatio)) || math.IsInf(float64(topKRatio), 0) {
		return fmt.Errorf("omnivoice: topKRatio must be finite and > 0 when classTemperature > 0")
	}
	if k < 1 {
		k = 1
	}
	if k > w.vocabSize {
		k = w.vocabSize
	}
	for row := 0; row < rows; row++ {
		off := row * w.vocabSize
		lp := logProbs[off : off+w.vocabSize]
		for i := range w.vocabScratch {
			w.vocabScratch[i] = float32(math.Inf(-1))
		}
		count := selectTopKStable(w.topVals, w.topIdx, lp, k)
		for i := 0; i < count; i++ {
			w.vocabScratch[w.topIdx[i]] = w.topVals[i]
		}
		rowNoise := sliceNoise(noise, off, w.vocabSize)
		if err = w.GumbelPerturbInto(w.vocabScratch2, w.vocabScratch, classTemperature, rowNoise); err != nil {
			return err
		}
		predTokens[row], _ = argmaxStable(w.vocabScratch2)
	}
	return nil
}

// PositionScoresInto applies OmniVoice's layer penalty to confidence scores.
// confidence and dst are flattened in [codebook][time] order.
func (w *SamplerWorkspace) PositionScoresInto(dst, confidence []float32, targetLen int, layerPenaltyFactor float32) error {
	rows, err := w.rowsFor(targetLen)
	if err != nil {
		return err
	}
	if len(dst) != rows || len(confidence) != rows {
		return fmt.Errorf("omnivoice: position-score shape mismatch")
	}
	for layer := 0; layer < w.codebooks; layer++ {
		penalty := float32(layer) * layerPenaltyFactor
		base := layer * targetLen
		for t := 0; t < targetLen; t++ {
			idx := base + t
			dst[idx] = confidence[idx] - penalty
		}
	}
	return nil
}

// ApplyConfidenceSelection updates masked tokens in place using OmniVoice's
// confidence-based top-k position selection. tokens must hold the current
// sampled tokens for one item in flattened [codebook][time] order. The returned
// indices alias workspace scratch and are valid only until the next workspace
// method call.
func (w *SamplerWorkspace) ApplyConfidenceSelection(tokens []int, predTokens []int, confidence []float32, targetLen, audioMaskID, k int, layerPenaltyFactor, positionTemperature float32, noise GumbelNoise) ([]int, error) {
	rows, err := w.rowsFor(targetLen)
	if err != nil {
		return nil, err
	}
	if len(tokens) != rows || len(predTokens) != rows || len(confidence) != rows {
		return nil, fmt.Errorf("omnivoice: selection shape mismatch")
	}
	if audioMaskID < 0 || audioMaskID >= w.vocabSize {
		return nil, fmt.Errorf("omnivoice: audioMaskID=%d outside [0,%d)", audioMaskID, w.vocabSize)
	}
	if k < 0 || k > rows {
		return nil, fmt.Errorf("omnivoice: k=%d outside [0,%d]", k, rows)
	}
	remaining := 0
	for _, tok := range tokens {
		if tok == audioMaskID {
			remaining++
		}
	}
	if k > remaining {
		return nil, fmt.Errorf("omnivoice: k=%d exceeds remaining masked positions=%d", k, remaining)
	}
	if err = w.PositionScoresInto(w.posScratch[:rows], confidence, targetLen, layerPenaltyFactor); err != nil {
		return nil, err
	}
	if positionTemperature > 0 {
		if err = w.GumbelPerturbInto(w.posScratch[:rows], w.posScratch[:rows], positionTemperature, noise); err != nil {
			return nil, err
		}
	}
	for i := range tokens {
		if tokens[i] != audioMaskID {
			w.posScratch[i] = float32(math.Inf(-1))
		}
	}
	count := selectTopKStable(w.topVals, w.topIdx, w.posScratch[:rows], k)
	for i := 0; i < count; i++ {
		tokens[w.topIdx[i]] = predTokens[w.topIdx[i]]
	}
	return w.topIdx[:count], nil
}

func validateNoise(noise GumbelNoise, want int) error {
	haveUniforms, haveGumbels := len(noise.Uniforms) != 0, len(noise.Gumbels) != 0
	if haveUniforms == haveGumbels {
		return fmt.Errorf("omnivoice: provide exactly one of uniforms or gumbels")
	}
	if haveUniforms {
		if len(noise.Uniforms) != want {
			return fmt.Errorf("omnivoice: uniform noise length mismatch")
		}
		for _, u := range noise.Uniforms {
			if math.IsNaN(float64(u)) || math.IsInf(float64(u), 0) || u < 0 || u > 1 {
				return fmt.Errorf("omnivoice: uniform noise must be finite and in [0,1]")
			}
		}
		return nil
	}
	if len(noise.Gumbels) != want {
		return fmt.Errorf("omnivoice: gumbel noise length mismatch")
	}
	for _, g := range noise.Gumbels {
		if math.IsNaN(float64(g)) || math.IsInf(float64(g), 0) {
			return fmt.Errorf("omnivoice: gumbel noise must be finite")
		}
	}
	return nil
}

func sliceNoise(noise GumbelNoise, offset, n int) GumbelNoise {
	if len(noise.Uniforms) != 0 {
		return GumbelNoise{Uniforms: noise.Uniforms[offset : offset+n]}
	}
	return GumbelNoise{Gumbels: noise.Gumbels[offset : offset+n]}
}

func logSoftmaxInto(dst, src []float32) {
	copy(dst, src)
	logSoftmaxInPlace(dst)
}

func logSoftmaxInPlace(row []float32) {
	if len(row) == 0 {
		return
	}
	maxVal := row[0]
	hasPosInf := math.IsInf(float64(maxVal), 1)
	posInfCount := 0
	if hasPosInf {
		posInfCount = 1
	}
	for _, v := range row[1:] {
		if v > maxVal {
			maxVal = v
		}
		if math.IsInf(float64(v), 1) {
			hasPosInf = true
			posInfCount++
		}
	}
	if hasPosInf {
		shared := float32(-math.Log(float64(posInfCount)))
		for i, v := range row {
			if math.IsInf(float64(v), 1) {
				row[i] = shared
			} else {
				row[i] = float32(math.Inf(-1))
			}
		}
		return
	}
	if math.IsInf(float64(maxVal), -1) {
		for i := range row {
			row[i] = float32(math.Inf(-1))
		}
		return
	}
	max64 := float64(maxVal)
	var sum float64
	for _, v := range row {
		sum += math.Exp(float64(v) - max64)
	}
	logZ := max64 + math.Log(sum)
	for i, v := range row {
		row[i] = v - float32(logZ)
	}
}

func argmaxStable(x []float32) (int, float32) {
	bestIdx := 0
	bestVal := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > bestVal {
			bestIdx, bestVal = i, x[i]
		}
	}
	return bestIdx, bestVal
}

func betterScore(aVal float32, aIdx int, bVal float32, bIdx int) bool {
	if aVal != bVal {
		return aVal > bVal
	}
	return aIdx < bIdx
}

func worseScore(aVal float32, aIdx int, bVal float32, bIdx int) bool {
	if aVal != bVal {
		return aVal < bVal
	}
	return aIdx > bIdx
}

func selectTopKStable(vals []float32, idx []int, scores []float32, k int) int {
	if k <= 0 || len(scores) == 0 {
		return 0
	}
	if k > len(scores) {
		k = len(scores)
	}
	n := 0
	for i, v := range scores {
		if n < k {
			vals[n] = v
			idx[n] = i
			siftUpWorst(vals, idx, n)
			n++
			continue
		}
		if betterScore(v, i, vals[0], idx[0]) {
			vals[0] = v
			idx[0] = i
			siftDownWorst(vals[:n], idx[:n], 0)
		}
	}
	for i := 1; i < n; i++ {
		vv, ii := vals[i], idx[i]
		j := i
		for j > 0 && betterScore(vv, ii, vals[j-1], idx[j-1]) {
			vals[j], idx[j] = vals[j-1], idx[j-1]
			j--
		}
		vals[j], idx[j] = vv, ii
	}
	return n
}

func siftUpWorst(vals []float32, idx []int, child int) {
	for child > 0 {
		parent := (child - 1) / 2
		if !worseScore(vals[child], idx[child], vals[parent], idx[parent]) {
			return
		}
		vals[child], vals[parent] = vals[parent], vals[child]
		idx[child], idx[parent] = idx[parent], idx[child]
		child = parent
	}
}

func siftDownWorst(vals []float32, idx []int, parent int) {
	n := len(vals)
	for {
		left := 2*parent + 1
		if left >= n {
			return
		}
		worst := left
		right := left + 1
		if right < n && worseScore(vals[right], idx[right], vals[left], idx[left]) {
			worst = right
		}
		if !worseScore(vals[worst], idx[worst], vals[parent], idx[parent]) {
			return
		}
		vals[parent], vals[worst] = vals[worst], vals[parent]
		idx[parent], idx[worst] = idx[worst], idx[parent]
		parent = worst
	}
}
