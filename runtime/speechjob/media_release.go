package speechjob

import (
	"context"
	"errors"
	"os"
	"time"
)

// ReleaseMedia removes retry-only source data after a terminal success or an
// explicit cancellation. Transcript, VTT, diarization and speaker checkpoints
// remain downloadable. Failed jobs retain media for retry. The manifest records
// release before deletion, so a crash can leave only safe redundant media;
// repeating this operation finishes cleanup.
func (s *Store) ReleaseMedia(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ready(); err != nil {
		return err
	}
	if s.busy {
		return ErrBusy
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	m, err := s.load(id)
	if err != nil {
		return err
	}
	if m.Status != Complete && m.Status != Cancelled {
		return ErrConfiguration
	}
	if !m.MediaReleased {
		m.MediaReleased = true
		m.Updated = time.Now().UTC()
		if err = s.save(m); err != nil {
			return errors.Join(ErrPersistence, err)
		}
	}
	files := []string{m.Input.File}
	for _, checkpoint := range m.Checkpoints {
		if checkpoint.Stage == "decode" {
			files = append(files, checkpoint.Blob.File)
		}
	}
	for _, file := range files {
		if err = ctx.Err(); err != nil {
			return err
		}
		if err = s.root.Remove(id + "/" + file); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	return s.syncDir(id)
}
