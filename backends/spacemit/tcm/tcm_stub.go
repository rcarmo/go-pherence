//go:build !linux || !riscv64

package tcm

import (
	"fmt"
	"runtime"
	"unsafe"
)

const (
	DevicePath = "/dev/tcm"
	BlockSize  = 393216
	BlockCount = 8
	TotalSize  = BlockSize * BlockCount
)

type Block struct{}

type TCM struct{}

func Open() (*TCM, error) {
	return nil, fmt.Errorf("tcm: unsupported on %s/%s (requires Linux/RISC-V K3)", runtime.GOOS, runtime.GOARCH)
}

func (t *TCM) Close() error { return nil }

func (t *TCM) Get(blockID int) (unsafe.Pointer, error) {
	return nil, fmt.Errorf("tcm: unsupported on this platform")
}

func (t *TCM) Release(blockID int) {}

func (t *TCM) Ptr(blockID int) unsafe.Pointer { return nil }

func (t *TCM) Slice(blockID int) []byte { return nil }

func IsAvailable() bool { return false }

func (t *TCM) Acquire(coreID int) error {
	return fmt.Errorf("tcm: unsupported on this platform")
}
