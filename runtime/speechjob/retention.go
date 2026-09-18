package speechjob

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"strings"
)

// RetainedJob exposes incomplete uploads and interrupted deletes as well as
// acknowledged jobs. Bytes includes manifests/orphans. Missing manifests are
// never queue entries. Corruption is reported without hiding the retained ID.
type RetainedJob struct {
	ID              string
	Bytes           int64
	ManifestPresent bool
	Deleting        bool
	Corrupt         bool
}

func (s *Store) Inventory() ([]RetainedJob, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.ready(); e != nil {
		return nil, e
	}
	if s.busy {
		return nil, ErrBusy
	}
	entries, e := fs.ReadDir(s.root.FS(), ".")
	if e != nil {
		return nil, e
	}
	out := []RetainedJob{}
	for _, entry := range entries {
		name, id := entry.Name(), entry.Name()
		deleting := strings.HasPrefix(name, ".deleted-")
		if deleting {
			id = strings.TrimPrefix(name, ".deleted-")
		}
		if !entry.IsDir() || !validID(id) {
			continue
		}
		r := RetainedJob{ID: id, Deleting: deleting}
		e = fs.WalkDir(s.root.FS(), name, func(p string, d fs.DirEntry, e error) error {
			if e != nil {
				return e
			}
			if d.Type()&os.ModeSymlink != 0 {
				return ErrCorrupt
			}
			if d.IsDir() {
				return nil
			}
			st, e := d.Info()
			if e != nil {
				return e
			}
			if !st.Mode().IsRegular() || st.Size() < 0 || st.Size() > s.limits.MaxBytes-r.Bytes {
				return ErrCorrupt
			}
			r.Bytes += st.Size()
			return nil
		})
		if e != nil {
			return nil, e
		}
		if !deleting {
			_, e = s.load(id)
			r.ManifestPresent = !errors.Is(e, os.ErrNotExist)
			r.Corrupt = e != nil && r.ManifestPresent
		}
		out = append(out, r)
	}
	return out, nil
}

// Delete is explicit irreversible retention cleanup. It refuses a busy store
// (including another job), first hides the ID via a durable same-root rename,
// then removes its data and syncs the root. A crash/failure may leave .deleted-ID;
// Inventory exposes it and repeating Delete(ID) finishes cleanup. No startup or
// automatic retention process deletes data. Open read handles may retain bytes
// until closed by the caller. Cancellation is checked before the rename only;
// after that point deletion completes synchronously or returns its IO error.
func (s *Store) Delete(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.ready(); e != nil {
		return e
	}
	if s.busy {
		return ErrBusy
	}
	if e := ctx.Err(); e != nil {
		return e
	}
	if !validID(id) {
		return ErrCorrupt
	}
	trash := ".deleted-" + id
	if st, e := s.root.Lstat(id); e == nil {
		if !st.IsDir() {
			return ErrCorrupt
		}
		if _, e = s.root.Lstat(trash); e == nil {
			return ErrCorrupt
		} else if !errors.Is(e, os.ErrNotExist) {
			return e
		}
		if e = s.root.Rename(id, trash); e != nil {
			return e
		}
		if e = s.syncDir("."); e != nil {
			return e
		}
		if e = s.hit("delete-renamed"); e != nil {
			return e
		}
	} else if !errors.Is(e, os.ErrNotExist) {
		return e
	} else {
		st, e = s.root.Lstat(trash)
		if errors.Is(e, os.ErrNotExist) {
			return nil
		}
		if e != nil {
			return e
		}
		if !st.IsDir() {
			return ErrCorrupt
		}
	}
	if e := s.root.RemoveAll(trash); e != nil {
		return e
	}
	return s.syncDir(".")
}
