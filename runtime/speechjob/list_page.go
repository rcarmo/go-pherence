package speechjob

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
)

// ListPage returns up to limit acknowledged manifests in lexicographic ID order,
// strictly after afterID. limit is 1..100; an empty afterID starts traversal. The
// next cursor is nonempty when more job directories exist (the next page can be
// empty if remaining directories have no acknowledged manifest). This is a live
// listing, not a transaction across pages: concurrent additions/deletions may
// change subsequent pages. Incomplete uploads are exposed through Inventory.
// Only one page of manifests is retained. Directory entries are bounded by the
// store contract. A corrupt acknowledged manifest still fails the page closed.
func (s *Store) ListPage(ctx context.Context, afterID string, limit int) ([]Manifest, string, error) {
	if limit < 1 || limit > 100 || afterID != "" && !validID(afterID) {
		return nil, "", fmt.Errorf("invalid speech job page")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.ready(); e != nil {
		return nil, "", e
	}
	if e := ctx.Err(); e != nil {
		return nil, "", e
	}
	entries, e := fs.ReadDir(s.root.FS(), ".")
	if e != nil {
		return nil, "", e
	}
	out := []Manifest{}
	last := ""
	for _, entry := range entries {
		if e = ctx.Err(); e != nil {
			return nil, "", e
		}
		if !entry.IsDir() || !validID(entry.Name()) || entry.Name() <= afterID {
			continue
		}
		if len(out) == limit {
			return out, last, nil
		}
		m, e := s.load(entry.Name())
		if errors.Is(e, os.ErrNotExist) {
			continue
		}
		if e != nil {
			return nil, "", e
		}
		out = append(out, clone(m))
		last = entry.Name()
	}
	return out, "", nil
}
