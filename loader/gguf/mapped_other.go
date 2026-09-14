//go:build !linux

package gguf

import "fmt"

// MapReadOnly returns an immutable in-memory copy of the entire GGUF file.
//
// The underlying file must remain immutable for the lifetime of any slices
// derived from the returned data. The returned release function has single
// ownership and must be called exactly once by the caller; wrap it if an
// idempotent closer is needed. All returned slices become invalid after
// release returns. Non-Linux platforms fall back to a whole-file ReadAt copy.
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
	buf := make([]byte, int(size))
	if _, err := g.f.ReadAt(buf, 0); err != nil {
		return nil, nil, fmt.Errorf("gguf: read: %w", err)
	}
	return buf, func() error { return nil }, nil
}
