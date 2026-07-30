package nvidia

import (
	"encoding/binary"
	"fmt"
	"sync"
	"unsafe"

	"github.com/rcarmo/go-pherence/internal/checked"
)

var fnMLXSelectedExpertPersistent CUfunction

// MLXSelectedExpertPersistentCandidate is an opt-in candidate launcher for the
// persistent selected-expert MLX projection PTX kernel.
//
// The caller must not call Run again, or Free the candidate, until the prior
// launch has completed on the default stream (for example via Sync, SyncErr, or
// SyncForTiming).
//
// Output is written as one contiguous row-major [len(workExperts), outDim]
// tile, where each row is weights[workExperts[i]] @ x.
type MLXSelectedExpertPersistentCandidate struct {
	mu sync.Mutex

	qPtrs      *Buffer
	scalePtrs  *Buffer
	biasPtrs   *Buffer
	workExpert *Buffer
	claim      *Buffer

	ptrCap  int
	workCap int

	hostQPtrs     []byte
	hostScalePtrs []byte
	hostBiasPtrs  []byte
}

func NewMLXSelectedExpertPersistentCandidate() *MLXSelectedExpertPersistentCandidate {
	return &MLXSelectedExpertPersistentCandidate{}
}

func (c *MLXSelectedExpertPersistentCandidate) Free() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.qPtrs != nil {
		c.qPtrs.Free()
		c.qPtrs = nil
	}
	if c.scalePtrs != nil {
		c.scalePtrs.Free()
		c.scalePtrs = nil
	}
	if c.biasPtrs != nil {
		c.biasPtrs.Free()
		c.biasPtrs = nil
	}
	if c.workExpert != nil {
		c.workExpert.Free()
		c.workExpert = nil
	}
	if c.claim != nil {
		c.claim.Free()
		c.claim = nil
	}
	c.ptrCap = 0
	c.workCap = 0
	c.hostQPtrs = nil
	c.hostScalePtrs = nil
	c.hostBiasPtrs = nil
}

func (c *MLXSelectedExpertPersistentCandidate) Run(out, x *DevBuf, weights []*GPUMLXWeight, workExperts []uint32) error {
	if c == nil {
		return fmt.Errorf("nil MLX selected expert candidate")
	}
	if len(workExperts) == 0 {
		return nil
	}
	if !SgemmReady() || fnMLXSelectedExpertPersistent == 0 {
		return fmt.Errorf("MLX selected expert persistent kernel not available")
	}
	inDim, outDim, groups, groupSize, err := validateMLXSelectedExpertNativeWeights(weights)
	if err != nil {
		return err
	}
	workLen := len(workExperts)
	totalOut, okTotal := checked.MulInt(workLen, outDim)
	if !okTotal {
		return fmt.Errorf("MLX selected expert output overflow work=%d outDim=%d", workLen, outDim)
	}
	if x == nil || out == nil || x.n < inDim || out.n < totalOut {
		return fmt.Errorf("invalid MLX selected expert buffers x=%d/%d out=%d/%d", devBufLen(x), inDim, devBufLen(out), totalOut)
	}
	if !fitsUint32(len(weights)) || !fitsUint32(workLen) || !fitsUint32(inDim) || !fitsUint32(outDim) || !fitsUint32(groups) || !fitsUint32(groupSize) || !fitsUint32(totalOut) {
		return fmt.Errorf("MLX selected expert dims exceed CUDA u32 interface")
	}
	for i, expert := range workExperts {
		if int(expert) >= len(weights) {
			return fmt.Errorf("work expert[%d]=%d outside [0,%d)", i, expert, len(weights))
		}
	}
	if err := x.ToGPU(); err != nil {
		return fmt.Errorf("MLX selected expert input ToGPU: %w", err)
	}
	if err := out.ToGPU(); err != nil {
		return fmt.Errorf("MLX selected expert output ToGPU: %w", err)
	}
	if x.gpu == nil || out.gpu == nil || x.gpu.Ptr == 0 || out.gpu.Ptr == 0 {
		return fmt.Errorf("MLX selected expert requires native GPU buffers")
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.ensurePointerCapacityLocked(len(weights)); err != nil {
		return err
	}
	if err := c.ensureWorkCapacityLocked(workLen); err != nil {
		return err
	}

	qHost := c.hostQPtrs[:len(weights)*8]
	sHost := c.hostScalePtrs[:len(weights)*8]
	bHost := c.hostBiasPtrs[:len(weights)*8]
	for i, w := range weights {
		binary.LittleEndian.PutUint64(qHost[i*8:(i+1)*8], uint64(w.QWeight.Ptr))
		binary.LittleEndian.PutUint64(sHost[i*8:(i+1)*8], uint64(w.Scales.Ptr))
		binary.LittleEndian.PutUint64(bHost[i*8:(i+1)*8], uint64(w.Biases.Ptr))
	}
	if err := c.qPtrs.UploadBytes(qHost); err != nil {
		return fmt.Errorf("upload MLX selected expert q pointer table: %w", err)
	}
	if err := c.scalePtrs.UploadBytes(sHost); err != nil {
		return fmt.Errorf("upload MLX selected expert scale pointer table: %w", err)
	}
	if err := c.biasPtrs.UploadBytes(bHost); err != nil {
		return fmt.Errorf("upload MLX selected expert bias pointer table: %w", err)
	}
	if err := c.workExpert.UploadUint32(workExperts); err != nil {
		return fmt.Errorf("upload MLX selected expert work indices: %w", err)
	}
	if err := ZeroFloat32Buffer(c.claim, 1); err != nil {
		return fmt.Errorf("zero MLX selected expert claim buffer: %w", err)
	}

	inDimU := uint32(inDim)
	outDimU := uint32(outDim)
	groupsU := uint32(groups)
	groupSizeU := uint32(groupSize)
	workLenU := uint32(workLen)
	activeExpertsU := uint32(len(weights))
	gridX := mlxSelectedExpertPersistentBlocks(totalOut)

	if err := LaunchKernel(fnMLXSelectedExpertPersistent, gridX, 1, 1, 256, 1, 1, 256*4,
		unsafe.Pointer(&x.gpu.Ptr),
		unsafe.Pointer(&c.qPtrs.Ptr),
		unsafe.Pointer(&c.scalePtrs.Ptr),
		unsafe.Pointer(&c.biasPtrs.Ptr),
		unsafe.Pointer(&c.workExpert.Ptr),
		unsafe.Pointer(&c.claim.Ptr),
		unsafe.Pointer(&out.gpu.Ptr),
		unsafe.Pointer(&inDimU),
		unsafe.Pointer(&outDimU),
		unsafe.Pointer(&groupsU),
		unsafe.Pointer(&groupSizeU),
		unsafe.Pointer(&workLenU),
		unsafe.Pointer(&activeExpertsU),
	); err != nil {
		return fmt.Errorf("launch MLX selected expert persistent: %w", err)
	}
	out.dev = GPU_DEVICE
	return nil
}

func (c *MLXSelectedExpertPersistentCandidate) ensurePointerCapacityLocked(n int) error {
	if n <= 0 {
		return fmt.Errorf("invalid MLX selected expert pointer count %d", n)
	}
	if c.ptrCap >= n && c.qPtrs != nil && c.scalePtrs != nil && c.biasPtrs != nil {
		return nil
	}
	if c.qPtrs != nil {
		c.qPtrs.Free()
		c.qPtrs = nil
	}
	if c.scalePtrs != nil {
		c.scalePtrs.Free()
		c.scalePtrs = nil
	}
	if c.biasPtrs != nil {
		c.biasPtrs.Free()
		c.biasPtrs = nil
	}
	bytes, ok := checked.MulInt(n, 8)
	if !ok {
		return fmt.Errorf("MLX selected expert pointer table overflow count=%d", n)
	}
	qPtrs, err := MallocBytes(bytes)
	if err != nil {
		return fmt.Errorf("alloc MLX selected expert q pointers: %w", err)
	}
	scalePtrs, err := MallocBytes(bytes)
	if err != nil {
		qPtrs.Free()
		return fmt.Errorf("alloc MLX selected expert scale pointers: %w", err)
	}
	biasPtrs, err := MallocBytes(bytes)
	if err != nil {
		qPtrs.Free()
		scalePtrs.Free()
		return fmt.Errorf("alloc MLX selected expert bias pointers: %w", err)
	}
	c.qPtrs = qPtrs
	c.scalePtrs = scalePtrs
	c.biasPtrs = biasPtrs
	c.ptrCap = n
	c.hostQPtrs = make([]byte, bytes)
	c.hostScalePtrs = make([]byte, bytes)
	c.hostBiasPtrs = make([]byte, bytes)
	return nil
}

func (c *MLXSelectedExpertPersistentCandidate) ensureWorkCapacityLocked(n int) error {
	if n <= 0 {
		return fmt.Errorf("invalid MLX selected expert work size %d", n)
	}
	if c.workCap < n || c.workExpert == nil {
		if c.workExpert != nil {
			c.workExpert.Free()
			c.workExpert = nil
		}
		buf, err := Malloc(n)
		if err != nil {
			return fmt.Errorf("alloc MLX selected expert work indices: %w", err)
		}
		c.workExpert = buf
		c.workCap = n
	}
	if c.claim == nil {
		buf, err := Malloc(1)
		if err != nil {
			return fmt.Errorf("alloc MLX selected expert claim buffer: %w", err)
		}
		c.claim = buf
	}
	return nil
}

func mlxSelectedExpertPersistentBlocks(totalOut int) uint32 {
	if totalOut <= 0 {
		return 1
	}
	blocks := totalOut
	if sms := SMCount(); sms > 0 {
		maxBlocks := sms * 4
		if maxBlocks < 1 {
			maxBlocks = 1
		}
		if blocks > maxBlocks {
			blocks = maxBlocks
		}
	}
	if blocks < 1 {
		blocks = 1
	}
	return uint32(blocks)
}

func validateMLXSelectedExpertNativeWeights(weights []*GPUMLXWeight) (inDim, outDim, groups, groupSize int, err error) {
	if len(weights) == 0 {
		return 0, 0, 0, 0, fmt.Errorf("empty MLX selected expert weight set")
	}
	for i, w := range weights {
		if err := validateNativeGPUMLXWeight(w); err != nil {
			return 0, 0, 0, 0, fmt.Errorf("weight %d: %w", i, err)
		}
		if i == 0 {
			inDim, outDim, groups, groupSize = w.InDim, w.OutDim, w.Groups, w.GroupSz
			continue
		}
		if w.InDim != inDim || w.OutDim != outDim || w.Groups != groups || w.GroupSz != groupSize {
			return 0, 0, 0, 0, fmt.Errorf("weight %d dims [%d,%d] groups=%d groupSize=%d mismatch base [%d,%d] groups=%d groupSize=%d", i, w.InDim, w.OutDim, w.Groups, w.GroupSz, inDim, outDim, groups, groupSize)
		}
	}
	return inDim, outDim, groups, groupSize, nil
}

func validateNativeGPUMLXWeight(w *GPUMLXWeight) error {
	if w == nil {
		return fmt.Errorf("nil MLX weight")
	}
	if w.InDim <= 0 || w.OutDim <= 0 || w.Groups <= 0 || w.GroupSz <= 0 || w.InDim%w.GroupSz != 0 || w.InDim%8 != 0 {
		return fmt.Errorf("invalid MLX dims in=%d out=%d groups=%d groupSize=%d", w.InDim, w.OutDim, w.Groups, w.GroupSz)
	}
	if w.GroupSz%8 != 0 {
		return fmt.Errorf("MLX group size %d must be divisible by 8 for native selected expert kernel", w.GroupSz)
	}
	if w.Groups != w.InDim/w.GroupSz {
		return fmt.Errorf("MLX group layout mismatch in=%d groups=%d groupSize=%d", w.InDim, w.Groups, w.GroupSz)
	}
	if w.QWeight == nil || w.Scales == nil || w.Biases == nil || w.QWeight.Ptr == 0 || w.Scales.Ptr == 0 || w.Biases.Ptr == 0 {
		return fmt.Errorf("missing native MLX buffers")
	}
	packed, okPacked := checked.MulInt(w.OutDim, w.InDim/8)
	scaleN, okScale := checked.MulInt(w.OutDim, w.Groups)
	if !okPacked || !okScale {
		return fmt.Errorf("MLX buffer size overflow")
	}
	packedBytes, errPacked := checkedByteSize(packed, -1)
	scaleBytes, errScale := checkedByteSize(scaleN, -1)
	if errPacked != nil || errScale != nil {
		return fmt.Errorf("MLX buffer byte size overflow")
	}
	if w.QWeight.Size < int(packedBytes) || w.Scales.Size < int(scaleBytes) || w.Biases.Size < int(scaleBytes) {
		return fmt.Errorf("short native MLX buffers q=%d/%d scales=%d/%d biases=%d/%d", w.QWeight.Size, packedBytes, w.Scales.Size, scaleBytes, w.Biases.Size, scaleBytes)
	}
	return nil
}

func devBufLen(b *DevBuf) int {
	if b == nil {
		return 0
	}
	return b.n
}
