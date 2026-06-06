package nvidia

import (
	"fmt"
	"unsafe"
)

var fnIdeogramCFGStepF32 CUfunction
var fnIdeogramLayerNormNoAffineF32 CUfunction
var fnIdeogramAdaLNTransformF32 CUfunction
var fnIdeogramGatedResidualF32 CUfunction
var fnIdeogramMRoPEF32 CUfunction
var fnIdeogramAttentionScoresF32 CUfunction
var fnIdeogramAttentionValuesF32 CUfunction
var fnIdeogramLatentDenormF32 CUfunction
var fnIdeogramRGBClampF32 CUfunction
var fnIdeogramUpsampleNearestF32 CUfunction
var fnIdeogramUnpatchifyF32 CUfunction

// IdeogramCFGStepBuffer fuses asymmetric CFG and FlowMatch Euler update on
// GPU-resident F32 buffers:
//
//	out = latents + sigma * (uncond + guidance * (cond - uncond))
//
// All buffers must hold at least n float32 elements.
func IdeogramCFGStepBuffer(out, latents, cond, uncond *Buffer, n int, guidance, sigma float32) error {
	if fnIdeogramCFGStepF32 == 0 || !megaModuleOK || out == nil || latents == nil || cond == nil || uncond == nil || n <= 0 || !fitsUint32(n) {
		return fmt.Errorf("invalid Ideogram CFG/step device buffers")
	}
	if _, err := checkedByteSize(n, out.Size); err != nil {
		return fmt.Errorf("invalid Ideogram CFG/step output buffer: %w", err)
	}
	if _, err := checkedByteSize(n, latents.Size); err != nil {
		return fmt.Errorf("invalid Ideogram CFG/step latent buffer: %w", err)
	}
	if _, err := checkedByteSize(n, cond.Size); err != nil {
		return fmt.Errorf("invalid Ideogram CFG/step conditional buffer: %w", err)
	}
	if _, err := checkedByteSize(n, uncond.Size); err != nil {
		return fmt.Errorf("invalid Ideogram CFG/step unconditional buffer: %w", err)
	}
	grid, ok := grid1DFor(n, 256)
	if !ok {
		return fmt.Errorf("invalid Ideogram CFG/step grid")
	}
	nn := uint32(n)
	return LaunchKernel(fnIdeogramCFGStepF32, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&latents.Ptr),
		unsafe.Pointer(&cond.Ptr),
		unsafe.Pointer(&uncond.Ptr),
		unsafe.Pointer(&out.Ptr),
		unsafe.Pointer(&guidance),
		unsafe.Pointer(&sigma),
		unsafe.Pointer(&nn))
}

// IdeogramLayerNormNoAffineBuffer computes row-wise non-affine LayerNorm on
// GPU-resident F32 buffers. x/out are row-major [rows, cols].
func IdeogramLayerNormNoAffineBuffer(out, x *Buffer, rows, cols int, eps float32) error {
	if fnIdeogramLayerNormNoAffineF32 == 0 || !megaModuleOK || out == nil || x == nil || rows <= 0 || cols <= 0 || !fitsUint32(rows) || !fitsUint32(cols) {
		return fmt.Errorf("invalid Ideogram LayerNorm device buffers")
	}
	n, ok := checkedMulInt(rows, cols)
	if !ok {
		return fmt.Errorf("Ideogram LayerNorm element count overflow rows=%d cols=%d", rows, cols)
	}
	if _, err := checkedByteSize(n, out.Size); err != nil {
		return fmt.Errorf("invalid Ideogram LayerNorm output buffer: %w", err)
	}
	if _, err := checkedByteSize(n, x.Size); err != nil {
		return fmt.Errorf("invalid Ideogram LayerNorm input buffer: %w", err)
	}
	rr := uint32(rows)
	cc := uint32(cols)
	return LaunchKernel(fnIdeogramLayerNormNoAffineF32, rr, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&x.Ptr),
		unsafe.Pointer(&out.Ptr),
		unsafe.Pointer(&rr),
		unsafe.Pointer(&cc),
		unsafe.Pointer(&eps))
}

// IdeogramLayerNormNoAffine computes row-wise non-affine LayerNorm using
// temporary device buffers. It is intended for correctness validation and early
// model wiring before the full Ideogram graph is GPU-resident.
func IdeogramLayerNormNoAffine(out, x []float32, rows, cols int, eps float32) error {
	n, ok := checkedMulInt(rows, cols)
	if !ok || rows <= 0 || cols <= 0 || len(out) < n || len(x) < n {
		return fmt.Errorf("invalid Ideogram LayerNorm host buffers out=%d/%d x=%d/%d", len(out), n, len(x), n)
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU not available")
	}
	EnsureContext()
	xBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc Ideogram LayerNorm input: %w", err)
	}
	defer xBuf.Free()
	outBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc Ideogram LayerNorm output: %w", err)
	}
	defer outBuf.Free()
	if err := xBuf.Upload(x[:n]); err != nil {
		return fmt.Errorf("upload Ideogram LayerNorm input: %w", err)
	}
	if err := IdeogramLayerNormNoAffineBuffer(outBuf, xBuf, rows, cols, eps); err != nil {
		return err
	}
	if err := outBuf.Download(out[:n]); err != nil {
		return fmt.Errorf("download Ideogram LayerNorm output: %w", err)
	}
	return nil
}

// IdeogramCFGStep computes the fused CFG+Euler update using temporary device
// buffers. It is a convenience/validation wrapper for the current Ideogram
// pipeline, which still returns DiT velocities as CPU slices.
func IdeogramCFGStep(out, latents, cond, uncond []float32, guidance, sigma float32) error {
	if len(out) == 0 || len(latents) != len(out) || len(cond) != len(out) || len(uncond) != len(out) {
		return fmt.Errorf("invalid Ideogram CFG/step host buffers out=%d latents=%d cond=%d uncond=%d", len(out), len(latents), len(cond), len(uncond))
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU not available")
	}
	EnsureContext()
	n := len(out)
	latBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc Ideogram CFG latent: %w", err)
	}
	defer latBuf.Free()
	condBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc Ideogram CFG conditional: %w", err)
	}
	defer condBuf.Free()
	uncondBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc Ideogram CFG unconditional: %w", err)
	}
	defer uncondBuf.Free()
	outBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc Ideogram CFG output: %w", err)
	}
	defer outBuf.Free()
	if err := latBuf.Upload(latents); err != nil {
		return fmt.Errorf("upload Ideogram CFG latent: %w", err)
	}
	if err := condBuf.Upload(cond); err != nil {
		return fmt.Errorf("upload Ideogram CFG conditional: %w", err)
	}
	if err := uncondBuf.Upload(uncond); err != nil {
		return fmt.Errorf("upload Ideogram CFG unconditional: %w", err)
	}
	if err := IdeogramCFGStepBuffer(outBuf, latBuf, condBuf, uncondBuf, n, guidance, sigma); err != nil {
		return err
	}
	if err := outBuf.Download(out); err != nil {
		return fmt.Errorf("download Ideogram CFG output: %w", err)
	}
	return nil
}

// F32RMSNormBuffer computes out = RMSNorm(x) * weight on GPU-resident F32
// buffers. It exposes the existing NVIDIA RMSNorm kernel for model code that
// uses low-level Buffer rather than DevBuf.
func F32RMSNormBuffer(out, x, weight *Buffer, n int, eps float32) error {
	if fnRmsNorm == 0 || !megaModuleOK || out == nil || x == nil || weight == nil || n <= 0 || !fitsUint32(n) {
		return fmt.Errorf("invalid F32 RMSNorm device buffers")
	}
	if _, err := checkedByteSize(n, out.Size); err != nil {
		return fmt.Errorf("invalid F32 RMSNorm output buffer: %w", err)
	}
	if _, err := checkedByteSize(n, x.Size); err != nil {
		return fmt.Errorf("invalid F32 RMSNorm input buffer: %w", err)
	}
	if _, err := checkedByteSize(n, weight.Size); err != nil {
		return fmt.Errorf("invalid F32 RMSNorm weight buffer: %w", err)
	}
	nn := uint32(n)
	return LaunchKernel(fnRmsNorm, 1, 1, 1, 256, 1, 1, 256*4,
		unsafe.Pointer(&x.Ptr),
		unsafe.Pointer(&weight.Ptr),
		unsafe.Pointer(&out.Ptr),
		unsafe.Pointer(&nn),
		unsafe.Pointer(&eps))
}

// F32RMSNorm computes out = RMSNorm(x) * weight through temporary device
// buffers. It is a correctness/wiring wrapper for early GPU conversion stages.
func F32RMSNorm(out, x, weight []float32, eps float32) error {
	if len(out) == 0 || len(x) != len(out) || len(weight) != len(out) {
		return fmt.Errorf("invalid F32 RMSNorm host buffers out=%d x=%d weight=%d", len(out), len(x), len(weight))
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU not available")
	}
	EnsureContext()
	n := len(out)
	xBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc F32 RMSNorm input: %w", err)
	}
	defer xBuf.Free()
	wBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc F32 RMSNorm weight: %w", err)
	}
	defer wBuf.Free()
	outBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc F32 RMSNorm output: %w", err)
	}
	defer outBuf.Free()
	if err := xBuf.Upload(x); err != nil {
		return fmt.Errorf("upload F32 RMSNorm input: %w", err)
	}
	if err := wBuf.Upload(weight); err != nil {
		return fmt.Errorf("upload F32 RMSNorm weight: %w", err)
	}
	if err := F32RMSNormBuffer(outBuf, xBuf, wBuf, n, eps); err != nil {
		return err
	}
	if err := outBuf.Download(out); err != nil {
		return fmt.Errorf("download F32 RMSNorm output: %w", err)
	}
	return nil
}

// IdeogramAdaLNTransformBuffer transforms [scale_msa, gate_msa, scale_mlp,
// gate_mlp] in-place on a GPU-resident F32 buffer, each slice length emb.
func IdeogramAdaLNTransformBuffer(mod *Buffer, emb int) error {
	if fnIdeogramAdaLNTransformF32 == 0 || !megaModuleOK || mod == nil || emb <= 0 || !fitsUint32(emb) {
		return fmt.Errorf("invalid Ideogram adaLN transform device buffer")
	}
	need, ok := checkedMulInt(4, emb)
	if !ok {
		return fmt.Errorf("Ideogram adaLN transform size overflow emb=%d", emb)
	}
	if _, err := checkedByteSize(need, mod.Size); err != nil {
		return fmt.Errorf("invalid Ideogram adaLN transform buffer: %w", err)
	}
	grid, ok := grid1DFor(emb, 256)
	if !ok {
		return fmt.Errorf("invalid Ideogram adaLN transform grid")
	}
	ee := uint32(emb)
	return LaunchKernel(fnIdeogramAdaLNTransformF32, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&mod.Ptr), unsafe.Pointer(&ee))
}

// IdeogramAdaLNTransform transforms a CPU modulation slice through temporary
// GPU storage. It is an early wiring/correctness wrapper before DiT blocks keep
// adaLN tensors GPU-resident.
func IdeogramAdaLNTransform(mod []float32, emb int) error {
	need, ok := checkedMulInt(4, emb)
	if !ok || emb <= 0 || len(mod) < need {
		return fmt.Errorf("invalid Ideogram adaLN transform host buffer len=%d want=%d", len(mod), need)
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU not available")
	}
	EnsureContext()
	buf, err := Malloc(need)
	if err != nil {
		return fmt.Errorf("alloc Ideogram adaLN transform: %w", err)
	}
	defer buf.Free()
	if err := buf.Upload(mod[:need]); err != nil {
		return fmt.Errorf("upload Ideogram adaLN transform: %w", err)
	}
	if err := IdeogramAdaLNTransformBuffer(buf, emb); err != nil {
		return err
	}
	if err := buf.Download(mod[:need]); err != nil {
		return fmt.Errorf("download Ideogram adaLN transform: %w", err)
	}
	return nil
}

// IdeogramGatedResidualBuffer computes hidden += gate * update in-place on
// GPU-resident F32 buffers.
func IdeogramGatedResidualBuffer(hidden, update, gate *Buffer, n int) error {
	if fnIdeogramGatedResidualF32 == 0 || !megaModuleOK || hidden == nil || update == nil || gate == nil || n <= 0 || !fitsUint32(n) {
		return fmt.Errorf("invalid Ideogram gated residual device buffers")
	}
	if _, err := checkedByteSize(n, hidden.Size); err != nil {
		return fmt.Errorf("invalid Ideogram gated residual hidden buffer: %w", err)
	}
	if _, err := checkedByteSize(n, update.Size); err != nil {
		return fmt.Errorf("invalid Ideogram gated residual update buffer: %w", err)
	}
	if _, err := checkedByteSize(n, gate.Size); err != nil {
		return fmt.Errorf("invalid Ideogram gated residual gate buffer: %w", err)
	}
	grid, ok := grid1DFor(n, 256)
	if !ok {
		return fmt.Errorf("invalid Ideogram gated residual grid")
	}
	nn := uint32(n)
	return LaunchKernel(fnIdeogramGatedResidualF32, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&hidden.Ptr), unsafe.Pointer(&update.Ptr), unsafe.Pointer(&gate.Ptr), unsafe.Pointer(&nn))
}

// IdeogramGatedResidual computes hidden += gate * update using temporary device
// buffers. Hidden is downloaded back in-place.
func IdeogramGatedResidual(hidden, update, gate []float32) error {
	if len(hidden) == 0 || len(update) != len(hidden) || len(gate) != len(hidden) {
		return fmt.Errorf("invalid Ideogram gated residual host buffers hidden=%d update=%d gate=%d", len(hidden), len(update), len(gate))
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU not available")
	}
	EnsureContext()
	n := len(hidden)
	hBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc Ideogram gated residual hidden: %w", err)
	}
	defer hBuf.Free()
	uBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc Ideogram gated residual update: %w", err)
	}
	defer uBuf.Free()
	gBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc Ideogram gated residual gate: %w", err)
	}
	defer gBuf.Free()
	if err := hBuf.Upload(hidden); err != nil {
		return fmt.Errorf("upload Ideogram gated residual hidden: %w", err)
	}
	if err := uBuf.Upload(update); err != nil {
		return fmt.Errorf("upload Ideogram gated residual update: %w", err)
	}
	if err := gBuf.Upload(gate); err != nil {
		return fmt.Errorf("upload Ideogram gated residual gate: %w", err)
	}
	if err := IdeogramGatedResidualBuffer(hBuf, uBuf, gBuf, n); err != nil {
		return err
	}
	if err := hBuf.Download(hidden); err != nil {
		return fmt.Errorf("download Ideogram gated residual hidden: %w", err)
	}
	return nil
}

// IdeogramMRoPEBuffer applies precomputed Ideogram MRoPE tables to a
// GPU-resident row-major [tokens, heads, headDim] F32 tensor in place. cos/sin
// are [tokens, headDim/2].
func IdeogramMRoPEBuffer(x, cosBuf, sinBuf *Buffer, tokens, heads, headDim int) error {
	if fnIdeogramMRoPEF32 == 0 || !megaModuleOK || x == nil || cosBuf == nil || sinBuf == nil || tokens <= 0 || heads <= 0 || headDim <= 0 || headDim%2 != 0 || !fitsUint32(tokens) || !fitsUint32(heads) || !fitsUint32(headDim) {
		return fmt.Errorf("invalid Ideogram MRoPE device buffers")
	}
	xLen, okX := checkedMulInt(tokens, heads)
	if okX {
		xLen, okX = checkedMulInt(xLen, headDim)
	}
	half := headDim / 2
	tableLen, okT := checkedMulInt(tokens, half)
	if !okX || !okT {
		return fmt.Errorf("Ideogram MRoPE size overflow tokens=%d heads=%d headDim=%d", tokens, heads, headDim)
	}
	if _, err := checkedByteSize(xLen, x.Size); err != nil {
		return fmt.Errorf("invalid Ideogram MRoPE x buffer: %w", err)
	}
	if _, err := checkedByteSize(tableLen, cosBuf.Size); err != nil {
		return fmt.Errorf("invalid Ideogram MRoPE cos buffer: %w", err)
	}
	if _, err := checkedByteSize(tableLen, sinBuf.Size); err != nil {
		return fmt.Errorf("invalid Ideogram MRoPE sin buffer: %w", err)
	}
	totalPairs, okPairs := checkedMulInt(tokens, heads)
	if okPairs {
		totalPairs, okPairs = checkedMulInt(totalPairs, half)
	}
	if !okPairs || !fitsUint32(totalPairs) {
		return fmt.Errorf("Ideogram MRoPE pair count overflow")
	}
	grid, ok := grid1DFor(totalPairs, 256)
	if !ok {
		return fmt.Errorf("invalid Ideogram MRoPE grid")
	}
	tt := uint32(tokens)
	hh := uint32(heads)
	hd := uint32(headDim)
	return LaunchKernel(fnIdeogramMRoPEF32, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&x.Ptr), unsafe.Pointer(&cosBuf.Ptr), unsafe.Pointer(&sinBuf.Ptr),
		unsafe.Pointer(&tt), unsafe.Pointer(&hh), unsafe.Pointer(&hd))
}

// IdeogramMRoPE applies precomputed MRoPE tables through temporary GPU buffers.
// It is a correctness/wiring wrapper for current CPU-slice DiT execution.
func IdeogramMRoPE(x, cosTable, sinTable []float32, tokens, heads, headDim int) error {
	xLen, okX := checkedMulInt(tokens, heads)
	if okX {
		xLen, okX = checkedMulInt(xLen, headDim)
	}
	half := headDim / 2
	tableLen, okT := checkedMulInt(tokens, half)
	if !okX || !okT || tokens <= 0 || heads <= 0 || headDim <= 0 || headDim%2 != 0 || len(x) < xLen || len(cosTable) < tableLen || len(sinTable) < tableLen {
		return fmt.Errorf("invalid Ideogram MRoPE host buffers x=%d/%d cos=%d/%d sin=%d/%d", len(x), xLen, len(cosTable), tableLen, len(sinTable), tableLen)
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU not available")
	}
	EnsureContext()
	xBuf, err := Malloc(xLen)
	if err != nil {
		return fmt.Errorf("alloc Ideogram MRoPE x: %w", err)
	}
	defer xBuf.Free()
	cosBuf, err := Malloc(tableLen)
	if err != nil {
		return fmt.Errorf("alloc Ideogram MRoPE cos: %w", err)
	}
	defer cosBuf.Free()
	sinBuf, err := Malloc(tableLen)
	if err != nil {
		return fmt.Errorf("alloc Ideogram MRoPE sin: %w", err)
	}
	defer sinBuf.Free()
	if err := xBuf.Upload(x[:xLen]); err != nil {
		return fmt.Errorf("upload Ideogram MRoPE x: %w", err)
	}
	if err := cosBuf.Upload(cosTable[:tableLen]); err != nil {
		return fmt.Errorf("upload Ideogram MRoPE cos: %w", err)
	}
	if err := sinBuf.Upload(sinTable[:tableLen]); err != nil {
		return fmt.Errorf("upload Ideogram MRoPE sin: %w", err)
	}
	if err := IdeogramMRoPEBuffer(xBuf, cosBuf, sinBuf, tokens, heads, headDim); err != nil {
		return err
	}
	if err := xBuf.Download(x[:xLen]); err != nil {
		return fmt.Errorf("download Ideogram MRoPE x: %w", err)
	}
	return nil
}

// SoftmaxRowsBuffer runs the existing NVIDIA row-softmax kernel over contiguous
// F32 rows. It is used by Ideogram full attention between score and value
// phases. in/out may be the same buffer.
func SoftmaxRowsBuffer(out, in *Buffer, rows, cols int) error {
	if softmaxRowsFn == 0 || !megaModuleOK || out == nil || in == nil || rows <= 0 || cols <= 0 || !fitsUint32(rows) || !fitsUint32(cols) || cols > 2048 {
		return fmt.Errorf("invalid row softmax device buffers")
	}
	total, ok := checkedMulInt(rows, cols)
	if !ok {
		return fmt.Errorf("row softmax size overflow rows=%d cols=%d", rows, cols)
	}
	if _, err := checkedByteSize(total, out.Size); err != nil {
		return fmt.Errorf("invalid row softmax output buffer: %w", err)
	}
	if _, err := checkedByteSize(total, in.Size); err != nil {
		return fmt.Errorf("invalid row softmax input buffer: %w", err)
	}
	cc := uint32(cols)
	return LaunchKernel(softmaxRowsFn, uint32(rows), 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&in.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&cc))
}

func IdeogramAttentionScoresBuffer(scores, q, k *Buffer, tokens, heads, headDim int, scale float32) error {
	if fnIdeogramAttentionScoresF32 == 0 || !megaModuleOK || scores == nil || q == nil || k == nil || tokens <= 0 || heads <= 0 || headDim <= 0 || !fitsUint32(tokens) || !fitsUint32(heads) || !fitsUint32(headDim) {
		return fmt.Errorf("invalid Ideogram attention score device buffers")
	}
	qLen, okQ := checkedMulInt(tokens, heads)
	if okQ {
		qLen, okQ = checkedMulInt(qLen, headDim)
	}
	scoreLen, okS := checkedMulInt(heads, tokens)
	if okS {
		scoreLen, okS = checkedMulInt(scoreLen, tokens)
	}
	if !okQ || !okS || !fitsUint32(scoreLen) {
		return fmt.Errorf("Ideogram attention score size overflow")
	}
	if _, err := checkedByteSize(qLen, q.Size); err != nil {
		return fmt.Errorf("invalid Ideogram attention q buffer: %w", err)
	}
	if _, err := checkedByteSize(qLen, k.Size); err != nil {
		return fmt.Errorf("invalid Ideogram attention k buffer: %w", err)
	}
	if _, err := checkedByteSize(scoreLen, scores.Size); err != nil {
		return fmt.Errorf("invalid Ideogram attention scores buffer: %w", err)
	}
	grid, ok := grid1DFor(scoreLen, 128)
	if !ok {
		return fmt.Errorf("invalid Ideogram attention score grid")
	}
	tt := uint32(tokens)
	hh := uint32(heads)
	hd := uint32(headDim)
	return LaunchKernel(fnIdeogramAttentionScoresF32, grid, 1, 1, 128, 1, 1, 0,
		unsafe.Pointer(&q.Ptr), unsafe.Pointer(&k.Ptr), unsafe.Pointer(&scores.Ptr),
		unsafe.Pointer(&tt), unsafe.Pointer(&hh), unsafe.Pointer(&hd), unsafe.Pointer(&scale))
}

func IdeogramAttentionValuesBuffer(out, probs, v *Buffer, tokens, heads, headDim int) error {
	if fnIdeogramAttentionValuesF32 == 0 || !megaModuleOK || out == nil || probs == nil || v == nil || tokens <= 0 || heads <= 0 || headDim <= 0 || !fitsUint32(tokens) || !fitsUint32(heads) || !fitsUint32(headDim) {
		return fmt.Errorf("invalid Ideogram attention value device buffers")
	}
	outLen, okOut := checkedMulInt(tokens, heads)
	if okOut {
		outLen, okOut = checkedMulInt(outLen, headDim)
	}
	probLen, okP := checkedMulInt(heads, tokens)
	if okP {
		probLen, okP = checkedMulInt(probLen, tokens)
	}
	if !okOut || !okP || !fitsUint32(outLen) {
		return fmt.Errorf("Ideogram attention value size overflow")
	}
	if _, err := checkedByteSize(outLen, out.Size); err != nil {
		return fmt.Errorf("invalid Ideogram attention output buffer: %w", err)
	}
	if _, err := checkedByteSize(outLen, v.Size); err != nil {
		return fmt.Errorf("invalid Ideogram attention v buffer: %w", err)
	}
	if _, err := checkedByteSize(probLen, probs.Size); err != nil {
		return fmt.Errorf("invalid Ideogram attention probs buffer: %w", err)
	}
	grid, ok := grid1DFor(outLen, 128)
	if !ok {
		return fmt.Errorf("invalid Ideogram attention value grid")
	}
	tt := uint32(tokens)
	hh := uint32(heads)
	hd := uint32(headDim)
	return LaunchKernel(fnIdeogramAttentionValuesF32, grid, 1, 1, 128, 1, 1, 0,
		unsafe.Pointer(&probs.Ptr), unsafe.Pointer(&v.Ptr), unsafe.Pointer(&out.Ptr),
		unsafe.Pointer(&tt), unsafe.Pointer(&hh), unsafe.Pointer(&hd))
}

// IdeogramFullAttention computes full non-causal attention for token-major
// [tokens, heads, headDim] Q/K/V through temporary device buffers.
func IdeogramFullAttention(out, q, k, v []float32, tokens, heads, headDim int, scale float32) error {
	outLen, okOut := checkedMulInt(tokens, heads)
	if okOut {
		outLen, okOut = checkedMulInt(outLen, headDim)
	}
	scoreLen, okS := checkedMulInt(heads, tokens)
	if okS {
		scoreLen, okS = checkedMulInt(scoreLen, tokens)
	}
	if !okOut || !okS || tokens <= 0 || heads <= 0 || headDim <= 0 || len(out) < outLen || len(q) < outLen || len(k) < outLen || len(v) < outLen {
		return fmt.Errorf("invalid Ideogram full attention host buffers out=%d/%d q=%d k=%d v=%d", len(out), outLen, len(q), len(k), len(v))
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU not available")
	}
	EnsureContext()
	qBuf, err := Malloc(outLen)
	if err != nil {
		return fmt.Errorf("alloc Ideogram attention q: %w", err)
	}
	defer qBuf.Free()
	kBuf, err := Malloc(outLen)
	if err != nil {
		return fmt.Errorf("alloc Ideogram attention k: %w", err)
	}
	defer kBuf.Free()
	vBuf, err := Malloc(outLen)
	if err != nil {
		return fmt.Errorf("alloc Ideogram attention v: %w", err)
	}
	defer vBuf.Free()
	scoreBuf, err := Malloc(scoreLen)
	if err != nil {
		return fmt.Errorf("alloc Ideogram attention scores: %w", err)
	}
	defer scoreBuf.Free()
	probBuf, err := Malloc(scoreLen)
	if err != nil {
		return fmt.Errorf("alloc Ideogram attention probs: %w", err)
	}
	defer probBuf.Free()
	outBuf, err := Malloc(outLen)
	if err != nil {
		return fmt.Errorf("alloc Ideogram attention out: %w", err)
	}
	defer outBuf.Free()
	if err := qBuf.Upload(q[:outLen]); err != nil {
		return fmt.Errorf("upload Ideogram attention q: %w", err)
	}
	if err := kBuf.Upload(k[:outLen]); err != nil {
		return fmt.Errorf("upload Ideogram attention k: %w", err)
	}
	if err := vBuf.Upload(v[:outLen]); err != nil {
		return fmt.Errorf("upload Ideogram attention v: %w", err)
	}
	if err := IdeogramAttentionScoresBuffer(scoreBuf, qBuf, kBuf, tokens, heads, headDim, scale); err != nil {
		return err
	}
	if err := SoftmaxRowsBuffer(probBuf, scoreBuf, heads*tokens, tokens); err != nil {
		return err
	}
	if err := IdeogramAttentionValuesBuffer(outBuf, probBuf, vBuf, tokens, heads, headDim); err != nil {
		return err
	}
	if err := outBuf.Download(out[:outLen]); err != nil {
		return fmt.Errorf("download Ideogram attention output: %w", err)
	}
	return nil
}

// F32SiLUBuffer computes out = silu(x) on GPU-resident F32 buffers.
func F32SiLUBuffer(out, x *Buffer, n int) error {
	if fnVecSilu == 0 || !megaModuleOK || out == nil || x == nil || n <= 0 || !fitsUint32(n) {
		return fmt.Errorf("invalid F32 SiLU device buffers")
	}
	if _, err := checkedByteSize(n, out.Size); err != nil {
		return fmt.Errorf("invalid F32 SiLU output buffer: %w", err)
	}
	if _, err := checkedByteSize(n, x.Size); err != nil {
		return fmt.Errorf("invalid F32 SiLU input buffer: %w", err)
	}
	grid, ok := grid1DFor(n, 256)
	if !ok {
		return fmt.Errorf("invalid F32 SiLU grid")
	}
	nn := uint32(n)
	return LaunchKernel(fnVecSilu, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&x.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&nn))
}

// F32MulBuffer computes out = a*b on GPU-resident F32 buffers.
func F32MulBuffer(out, a, b *Buffer, n int) error {
	if fnVecMul == 0 || !megaModuleOK || out == nil || a == nil || b == nil || n <= 0 || !fitsUint32(n) {
		return fmt.Errorf("invalid F32 Mul device buffers")
	}
	if _, err := checkedByteSize(n, out.Size); err != nil {
		return fmt.Errorf("invalid F32 Mul output buffer: %w", err)
	}
	if _, err := checkedByteSize(n, a.Size); err != nil {
		return fmt.Errorf("invalid F32 Mul input A buffer: %w", err)
	}
	if _, err := checkedByteSize(n, b.Size); err != nil {
		return fmt.Errorf("invalid F32 Mul input B buffer: %w", err)
	}
	grid, ok := grid1DFor(n, 256)
	if !ok {
		return fmt.Errorf("invalid F32 Mul grid")
	}
	nn := uint32(n)
	return LaunchKernel(fnVecMul, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&a.Ptr), unsafe.Pointer(&b.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&nn))
}

// F32SiLU computes out = silu(x) through temporary device buffers.
func F32SiLU(out, x []float32) error {
	if len(out) == 0 || len(x) != len(out) {
		return fmt.Errorf("invalid F32 SiLU host buffers out=%d x=%d", len(out), len(x))
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU not available")
	}
	n := len(out)
	xBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc F32 SiLU input: %w", err)
	}
	defer xBuf.Free()
	outBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc F32 SiLU output: %w", err)
	}
	defer outBuf.Free()
	if err := xBuf.Upload(x); err != nil {
		return fmt.Errorf("upload F32 SiLU input: %w", err)
	}
	if err := F32SiLUBuffer(outBuf, xBuf, n); err != nil {
		return err
	}
	if err := outBuf.Download(out); err != nil {
		return fmt.Errorf("download F32 SiLU output: %w", err)
	}
	return nil
}

// F32Mul computes out = a*b through temporary device buffers.
func F32Mul(out, a, b []float32) error {
	if len(out) == 0 || len(a) != len(out) || len(b) != len(out) {
		return fmt.Errorf("invalid F32 Mul host buffers out=%d a=%d b=%d", len(out), len(a), len(b))
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU not available")
	}
	n := len(out)
	aBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc F32 Mul input A: %w", err)
	}
	defer aBuf.Free()
	bBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc F32 Mul input B: %w", err)
	}
	defer bBuf.Free()
	outBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc F32 Mul output: %w", err)
	}
	defer outBuf.Free()
	if err := aBuf.Upload(a); err != nil {
		return fmt.Errorf("upload F32 Mul input A: %w", err)
	}
	if err := bBuf.Upload(b); err != nil {
		return fmt.Errorf("upload F32 Mul input B: %w", err)
	}
	if err := F32MulBuffer(outBuf, aBuf, bBuf, n); err != nil {
		return err
	}
	if err := outBuf.Download(out); err != nil {
		return fmt.Errorf("download F32 Mul output: %w", err)
	}
	return nil
}

// F32SiLUMul computes out = silu(a)*b through temporary device buffers.
func F32SiLUMul(out, a, b []float32) error {
	if len(out) == 0 || len(a) != len(out) || len(b) != len(out) {
		return fmt.Errorf("invalid F32 SiLU*Mul host buffers out=%d a=%d b=%d", len(out), len(a), len(b))
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU not available")
	}
	n := len(out)
	aBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc F32 SiLU*Mul input A: %w", err)
	}
	defer aBuf.Free()
	bBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc F32 SiLU*Mul input B: %w", err)
	}
	defer bBuf.Free()
	outBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc F32 SiLU*Mul output: %w", err)
	}
	defer outBuf.Free()
	if err := aBuf.Upload(a); err != nil {
		return fmt.Errorf("upload F32 SiLU*Mul input A: %w", err)
	}
	if err := bBuf.Upload(b); err != nil {
		return fmt.Errorf("upload F32 SiLU*Mul input B: %w", err)
	}
	if err := F32SiLUMulBuffer(outBuf, aBuf, bBuf, n); err != nil {
		return err
	}
	if err := outBuf.Download(out); err != nil {
		return fmt.Errorf("download F32 SiLU*Mul output: %w", err)
	}
	return nil
}

// IdeogramLatentDenormBuffer applies per-channel latent denormalization in
// place on a GPU-resident F32 buffer laid out [tokens, channels].
func IdeogramLatentDenormBuffer(x, scale, shift *Buffer, n, channels int) error {
	loadMegaModule()
	if fnIdeogramLatentDenormF32 == 0 || !megaModuleOK || x == nil || scale == nil || shift == nil || n <= 0 || channels <= 0 || !fitsUint32(n) || !fitsUint32(channels) {
		return fmt.Errorf("invalid Ideogram latent denorm device buffers")
	}
	if _, err := checkedByteSize(n, x.Size); err != nil {
		return fmt.Errorf("invalid Ideogram latent denorm data buffer: %w", err)
	}
	if _, err := checkedByteSize(channels, scale.Size); err != nil {
		return fmt.Errorf("invalid Ideogram latent denorm scale buffer: %w", err)
	}
	if _, err := checkedByteSize(channels, shift.Size); err != nil {
		return fmt.Errorf("invalid Ideogram latent denorm shift buffer: %w", err)
	}
	grid, ok := grid1DFor(n, 256)
	if !ok {
		return fmt.Errorf("invalid Ideogram latent denorm grid")
	}
	nn := uint32(n)
	cc := uint32(channels)
	return LaunchKernel(fnIdeogramLatentDenormF32, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&x.Ptr),
		unsafe.Pointer(&scale.Ptr),
		unsafe.Pointer(&shift.Ptr),
		unsafe.Pointer(&nn),
		unsafe.Pointer(&cc))
}

// IdeogramLatentDenorm applies per-channel latent denormalization in place on
// host F32 data by staging through the NVIDIA kernel.
func IdeogramLatentDenorm(x, scale, shift []float32, channels int) error {
	if channels <= 0 || len(x) == 0 || len(x)%channels != 0 || len(scale) < channels || len(shift) < channels {
		return fmt.Errorf("invalid Ideogram latent denorm host buffers x=%d channels=%d scale=%d shift=%d", len(x), channels, len(scale), len(shift))
	}
	xBuf, err := Malloc(len(x))
	if err != nil {
		return err
	}
	defer xBuf.Free()
	sBuf, err := Malloc(channels)
	if err != nil {
		return err
	}
	defer sBuf.Free()
	shBuf, err := Malloc(channels)
	if err != nil {
		return err
	}
	defer shBuf.Free()
	if err := xBuf.Upload(x); err != nil {
		return err
	}
	if err := sBuf.Upload(scale[:channels]); err != nil {
		return err
	}
	if err := shBuf.Upload(shift[:channels]); err != nil {
		return err
	}
	if err := IdeogramLatentDenormBuffer(xBuf, sBuf, shBuf, len(x), channels); err != nil {
		return err
	}
	return xBuf.Download(x)
}

// IdeogramRGBClamp converts CHW RGB F32 values in [-1,1] to interleaved RGB F32
// values in [0,255] through the NVIDIA kernel.
func IdeogramRGBClamp(out, in []float32, hw int) error {
	loadMegaModule()
	if fnIdeogramRGBClampF32 == 0 || !megaModuleOK || hw <= 0 || len(in) < 3*hw || len(out) < 3*hw || !fitsUint32(hw) {
		return fmt.Errorf("invalid Ideogram RGB clamp buffers out=%d in=%d hw=%d", len(out), len(in), hw)
	}
	inBuf, err := Malloc(3 * hw)
	if err != nil {
		return err
	}
	defer inBuf.Free()
	outBuf, err := Malloc(3 * hw)
	if err != nil {
		return err
	}
	defer outBuf.Free()
	if err := inBuf.Upload(in[:3*hw]); err != nil {
		return err
	}
	grid, ok := grid1DFor(3*hw, 256)
	if !ok {
		return fmt.Errorf("invalid Ideogram RGB clamp grid")
	}
	h := uint32(hw)
	if err := LaunchKernel(fnIdeogramRGBClampF32, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&outBuf.Ptr),
		unsafe.Pointer(&inBuf.Ptr),
		unsafe.Pointer(&h)); err != nil {
		return err
	}
	return outBuf.Download(out[:3*hw])
}

// IdeogramUpsampleNearest upsamples a CHW F32 feature map by nearest-neighbour.
func IdeogramUpsampleNearest(out, in []float32, c, h, w, factor int) error {
	loadMegaModule()
	if fnIdeogramUpsampleNearestF32 == 0 || !megaModuleOK || c <= 0 || h <= 0 || w <= 0 || factor <= 0 || !fitsUint32(c) || !fitsUint32(h) || !fitsUint32(w) || !fitsUint32(factor) {
		return fmt.Errorf("invalid Ideogram upsample dims c=%d h=%d w=%d factor=%d", c, h, w, factor)
	}
	inN, okIn := checkedMulInt(c, h*w)
	outH, okH := checkedMulInt(h, factor)
	outW, okW := checkedMulInt(w, factor)
	outN, okOut := checkedMulInt(c, outH*outW)
	if !okIn || !okH || !okW || !okOut || len(in) < inN || len(out) < outN || !fitsUint32(outN) {
		return fmt.Errorf("invalid Ideogram upsample buffers out=%d/%d in=%d/%d", len(out), outN, len(in), inN)
	}
	inBuf, err := Malloc(inN)
	if err != nil {
		return err
	}
	defer inBuf.Free()
	outBuf, err := Malloc(outN)
	if err != nil {
		return err
	}
	defer outBuf.Free()
	if err := inBuf.Upload(in[:inN]); err != nil {
		return err
	}
	grid, ok := grid1DFor(outN, 256)
	if !ok {
		return fmt.Errorf("invalid Ideogram upsample grid")
	}
	cc, hh, ww, ff, total := uint32(c), uint32(h), uint32(w), uint32(factor), uint32(outN)
	if err := LaunchKernel(fnIdeogramUpsampleNearestF32, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&outBuf.Ptr), unsafe.Pointer(&inBuf.Ptr), unsafe.Pointer(&cc), unsafe.Pointer(&hh), unsafe.Pointer(&ww), unsafe.Pointer(&ff), unsafe.Pointer(&total)); err != nil {
		return err
	}
	return outBuf.Download(out[:outN])
}

// IdeogramUnpatchify converts token-major patchified latents to a CHW F32 map.
func IdeogramUnpatchify(out, tokens []float32, gridH, gridW, inChannels, latentChannels, patchH, patchW int) error {
	loadMegaModule()
	H, okH := checkedMulInt(gridH, patchH)
	W, okW := checkedMulInt(gridW, patchW)
	HW, okHW := checkedMulInt(H, W)
	outN, okOut := checkedMulInt(latentChannels, HW)
	tokN, okTok := checkedMulInt(gridH*gridW, inChannels)
	if fnIdeogramUnpatchifyF32 == 0 || !megaModuleOK || gridH <= 0 || gridW <= 0 || inChannels <= 0 || latentChannels <= 0 || patchH <= 0 || patchW <= 0 || !okH || !okW || !okHW || !okOut || !okTok || len(out) < outN || len(tokens) < tokN || !fitsUint32(gridH) || !fitsUint32(gridW) || !fitsUint32(inChannels) || !fitsUint32(latentChannels) || !fitsUint32(patchH) || !fitsUint32(patchW) || !fitsUint32(outN) {
		return fmt.Errorf("invalid Ideogram unpatchify buffers out=%d/%d tokens=%d/%d", len(out), outN, len(tokens), tokN)
	}
	outBuf, err := Malloc(outN)
	if err != nil {
		return err
	}
	defer outBuf.Free()
	tokBuf, err := Malloc(tokN)
	if err != nil {
		return err
	}
	defer tokBuf.Free()
	if err := tokBuf.Upload(tokens[:tokN]); err != nil {
		return err
	}
	grid, ok := grid1DFor(outN, 256)
	if !ok {
		return fmt.Errorf("invalid Ideogram unpatchify grid")
	}
	gh, gw, ic, lc, ph, pw, total := uint32(gridH), uint32(gridW), uint32(inChannels), uint32(latentChannels), uint32(patchH), uint32(patchW), uint32(outN)
	if err := LaunchKernel(fnIdeogramUnpatchifyF32, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&outBuf.Ptr), unsafe.Pointer(&tokBuf.Ptr), unsafe.Pointer(&gh), unsafe.Pointer(&gw), unsafe.Pointer(&ic), unsafe.Pointer(&lc), unsafe.Pointer(&ph), unsafe.Pointer(&pw), unsafe.Pointer(&total)); err != nil {
		return err
	}
	return outBuf.Download(out[:outN])
}
