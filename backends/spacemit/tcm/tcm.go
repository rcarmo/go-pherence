// Package tcm provides direct access to SpacemiT TCM (Tightly Coupled Memory)
// on the K3 SoC. TCM is 3MB of on-chip SRAM divided into 8 x 384KB blocks,
// accessible via mmap of /dev/tcm.
package tcm

import (
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"
)

const (
	DevicePath = "/dev/tcm"
	BlockSize  = 393216 // 384KB per block
	BlockCount = 8
	TotalSize  = BlockSize * BlockCount // 3MB
)

// Block represents a single 384KB TCM SRAM block.
type Block struct {
	mu     sync.Mutex
	ptr    unsafe.Pointer
	refcnt atomic.Int64
	id     int
}

// TCM manages access to the on-chip SRAM blocks.
type TCM struct {
	fd     int
	base   uintptr
	data   []byte // mmap'd region
	blocks [BlockCount]Block
}

// Open initializes TCM by opening /dev/tcm and mmap'ing the full 3MB region.
func Open() (*TCM, error) {
	fd, err := syscall.Open(DevicePath, syscall.O_RDWR|syscall.O_SYNC, 0)
	if err != nil {
		return nil, fmt.Errorf("tcm: open %s: %w", DevicePath, err)
	}

	data, err := syscall.Mmap(fd, 0, TotalSize, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		syscall.Close(fd)
		return nil, fmt.Errorf("tcm: mmap: %w", err)
	}

	t := &TCM{
		fd:   fd,
		base: uintptr(unsafe.Pointer(&data[0])),
		data: data,
	}
	for i := range t.blocks {
		t.blocks[i].id = i
		t.blocks[i].ptr = unsafe.Pointer(t.base + uintptr(i*BlockSize))
	}
	return t, nil
}

// Close unmaps and closes the TCM device.
func (t *TCM) Close() error {
	if t.data != nil {
		syscall.Munmap(t.data)
		t.data = nil
	}
	if t.fd > 0 {
		syscall.Close(t.fd)
		t.fd = 0
	}
	return nil
}

// Get acquires a TCM block, incrementing its reference count.
// Returns a pointer to the 384KB SRAM region.
func (t *TCM) Get(blockID int) (unsafe.Pointer, error) {
	if blockID < 0 || blockID >= BlockCount {
		return nil, fmt.Errorf("tcm: invalid block %d", blockID)
	}
	b := &t.blocks[blockID]
	b.mu.Lock()
	b.refcnt.Add(1)
	ptr := b.ptr
	b.mu.Unlock()
	return ptr, nil
}

// Release decrements the reference count for a block.
func (t *TCM) Release(blockID int) {
	if blockID < 0 || blockID >= BlockCount {
		return
	}
	t.blocks[blockID].refcnt.Add(-1)
}

// Ptr returns the base pointer for a block (no locking).
func (t *TCM) Ptr(blockID int) unsafe.Pointer {
	if blockID < 0 || blockID >= BlockCount {
		return nil
	}
	return t.blocks[blockID].ptr
}

// Slice returns a byte slice view of a TCM block.
func (t *TCM) Slice(blockID int) []byte {
	offset := blockID * BlockSize
	return t.data[offset : offset+BlockSize]
}

// IsAvailable returns true if /dev/tcm exists and is accessible.
func IsAvailable() bool {
	_, err := os.Stat(DevicePath)
	return err == nil
}
