//go:build linux

package gguf

import (
	"fmt"
	"syscall"
)

// MapReadOnly maps the entire GGUF file read-only and returns a byte slice
// backed by the original file contents.
//
// The underlying file must remain immutable for the lifetime of any slices
// derived from the returned data. The returned release function has single
// ownership and must be called exactly once by the caller; wrap it if an
// idempotent closer is needed. All returned slices become invalid after
// release returns. On Linux the mapping remains valid after g.Close() because
// the kernel keeps the mapping alive after the file descriptor is closed.
func (g *GGUF) MapReadOnly() ([]byte, func() error, error) {
	if g == nil || g.f == nil {
		return nil, nil, fmt.Errorf("gguf: reader is nil")
	}
	fi, err := g.f.Stat()
	if err != nil {
		return nil, nil, fmt.Errorf("gguf: stat: %w", err)
	}
	size := fi.Size()
	if size <= 0 {
		return nil, nil, fmt.Errorf("gguf: invalid mapped file size %d", size)
	}
	if size > int64(ggufIntLimit()) {
		return nil, nil, fmt.Errorf("gguf: mapped file size %d exceeds int", size)
	}
	mapped, err := syscall.Mmap(int(g.f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		return nil, nil, fmt.Errorf("gguf: mmap: %w", err)
	}
	return mapped, func() error {
		if err := syscall.Munmap(mapped); err != nil {
			return fmt.Errorf("gguf: munmap: %w", err)
		}
		return nil
	}, nil
}
