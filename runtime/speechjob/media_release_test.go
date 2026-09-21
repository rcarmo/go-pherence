package speechjob

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestReleaseCompletedMediaPreservesPublishedArtifacts(t *testing.T) {
	store, root := openTest(t)
	job := createTest(t, store)
	stages := []Stage{
		stage("decode", func(context.Context, *Input, io.Writer) error { return nil }),
		stage("transcript", func(_ context.Context, _ *Input, out io.Writer) error {
			_, err := io.WriteString(out, "published transcript")
			return err
		}),
	}
	job, err := store.Run(context.Background(), job.ID, config, stages, nil)
	if err != nil || job.Status != Complete {
		t.Fatal(job, err)
	}
	if err = store.ReleaseMedia(context.Background(), job.ID); err != nil {
		t.Fatal(err)
	}
	if err = store.ReleaseMedia(context.Background(), job.ID); err != nil {
		t.Fatal("release not idempotent", err)
	}
	stored, err := store.Get(job.ID)
	if err != nil || !stored.MediaReleased {
		t.Fatal(stored, err)
	}
	if _, err = os.Stat(filepath.Join(root, job.ID, "input")); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("input retained", err)
	}
	if _, err = os.Stat(filepath.Join(root, job.ID, job.Checkpoints[0].Blob.File)); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("decode retained", err)
	}
	reader, err := store.OpenCheckpoint(context.Background(), job.ID, "transcript")
	if err != nil {
		t.Fatal(err)
	}
	data, readErr := io.ReadAll(reader)
	closeErr := reader.Close()
	if readErr != nil || closeErr != nil || string(data) != "published transcript" {
		t.Fatal(string(data), readErr, closeErr)
	}
	if _, err = store.Run(context.Background(), job.ID, config, stages, nil); !errors.Is(err, ErrConfiguration) {
		t.Fatal("released job reran", err)
	}
	if err = store.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := Open(root, Limits{MaxJobs: 10, MaxUploadBytes: 1024, MaxArtifactBytes: 1024, MaxBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	stored, err = reopened.Get(job.ID)
	if err != nil || !stored.MediaReleased {
		t.Fatal("release not durable", stored, err)
	}
	reader, err = reopened.OpenCheckpoint(context.Background(), job.ID, "transcript")
	if err != nil {
		t.Fatal("published artifact lost after reopen", err)
	}
	reader.Close()
}

func TestReleaseMediaRejectsRetryableFailure(t *testing.T) {
	store, root := openTest(t)
	job := createTest(t, store)
	failure := errors.New("fixture failure")
	job, err := store.Run(context.Background(), job.ID, config, []Stage{stage("decode", func(context.Context, *Input, io.Writer) error { return failure })}, nil)
	if !errors.Is(err, failure) || job.Status != Failed {
		t.Fatal(job, err)
	}
	if err = store.ReleaseMedia(context.Background(), job.ID); !errors.Is(err, ErrConfiguration) {
		t.Fatal("failed media released", err)
	}
	if _, err = os.Stat(filepath.Join(root, job.ID, "input")); err != nil {
		t.Fatal("retry input removed", err)
	}
}
