package expertstream

import (
	"fmt"
	"math"
	"unsafe"

	"golang.org/x/sys/unix"
)

type slot struct {
	index   int
	raw     []byte
	buf     []byte
	loaded  bool
	key     uint64
	span    int64
	lastUse uint64
}

func allocAligned(size, alignment int64) ([]byte, []byte, error) {
	if size <= 0 {
		return nil, nil, fmt.Errorf("size must be positive")
	}
	if !isPowerOfTwo(alignment) {
		return nil, nil, fmt.Errorf("alignment must be a positive power of two")
	}
	rawLen64, err := checkedAdd(size, alignment-1)
	if err != nil {
		return nil, nil, fmt.Errorf("aligned allocation overflow")
	}
	if rawLen64 > math.MaxInt {
		return nil, nil, fmt.Errorf("aligned allocation too large")
	}
	raw, err := unix.Mmap(-1, 0, int(rawLen64), unix.PROT_READ|unix.PROT_WRITE, unix.MAP_PRIVATE|unix.MAP_ANONYMOUS)
	if err != nil {
		return nil, nil, fmt.Errorf("mmap: %w", err)
	}
	base := uintptr(unsafe.Pointer(&raw[0]))
	aligned := alignPtr(base, uintptr(alignment))
	delta := int(aligned - base)
	if delta < 0 || delta+int(size) > len(raw) {
		_ = unix.Munmap(raw)
		return nil, nil, fmt.Errorf("aligned allocation bounds")
	}
	return raw, raw[delta : delta+int(size)], nil
}

func freeAligned(raw []byte) error {
	if len(raw) == 0 {
		return nil
	}
	return unix.Munmap(raw)
}

func alignPtr(v, alignment uintptr) uintptr {
	mask := alignment - 1
	return (v + mask) &^ mask
}
