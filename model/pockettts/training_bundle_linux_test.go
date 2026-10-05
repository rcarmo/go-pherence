//go:build linux

package pockettts

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
)

// Opaque association fixtures have valid magic/version/step only. They are NOT
// decoder-valid trainable checkpoints and never call a model/trainer/codec.
func bundleModelFixture(step uint64) []byte {
	data := make([]byte, trainingBundlePrefixBytes+13)
	copy(data, fullTrainingBinaryMagic[:])
	binary.LittleEndian.PutUint32(data[16:20], 1)
	binary.LittleEndian.PutUint64(data[20:28], step)
	copy(data[52:], "opaque-not-tts")
	return data
}
func bundleLimitsFixture() TrainingBundleLimits {
	return TrainingBundleLimits{MaxModelBytes: 1024, MaxSidecarBytes: 8192}
}
func bundlePublishFixture(t *testing.T, dir, generation string, step uint64, run *TrainingRunState) {
	t.Helper()
	if _, err := PublishTrainingBundle(dir, generation, bytes.NewReader(bundleModelFixture(step)), step, run, bundleLimitsFixture()); err != nil {
		t.Fatal(err)
	}
}
func bundleReadFixture(t *testing.T, dir string, step uint64) {
	t.Helper()
	b, err := OpenTrainingBundle(dir, runIdentityFixture(), bundleLimitsFixture())
	if err != nil {
		t.Fatal(err)
	}
	raw, err := io.ReadAll(b.Model)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, bundleModelFixture(step)) || b.Checkpoint.Step != step {
		t.Fatal("bundle pair/prefix changed")
	}
	got, _, _ := b.Run.Progress()
	if got != step {
		t.Fatal("run/model step mismatch")
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
}
func bundleWriteFixture(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
}
func bundleBytesFixture(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestTrainingBundleRoundTrip(t *testing.T) {
	dir := t.TempDir()
	run := newRunFixture(t, 3)
	before := snapshotRunFixture(t, run)
	bundlePublishFixture(t, dir, "g0", 0, run)
	bundleReadFixture(t, dir, 0)
	if !bytes.Equal(before, snapshotRunFixture(t, run)) {
		t.Fatal("publisher mutated run")
	}
	oldModel := bundleBytesFixture(t, filepath.Join(dir, "g0.model.bin"))
	oldSidecar := bundleBytesFixture(t, filepath.Join(dir, "g0.run.json"))
	rowAdvance(t, run, 2)
	bundlePublishFixture(t, dir, "g1", 1, run)
	bundleReadFixture(t, dir, 1)
	if !bytes.Equal(oldModel, bundleBytesFixture(t, filepath.Join(dir, "g0.model.bin"))) || !bytes.Equal(oldSidecar, bundleBytesFixture(t, filepath.Join(dir, "g0.run.json"))) {
		t.Fatal("previous immutable generation changed")
	}
	if _, err := os.Stat(filepath.Join(dir, "g1.pointer.tmp")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("pointer temp should be renamed: %v", err)
	}
}
func TestTrainingBundleCollisionAndAdmission(t *testing.T) {
	dir := t.TempDir()
	run := newRunFixture(t, 2)
	bundlePublishFixture(t, dir, "old", 0, run)
	pointer := bundleBytesFixture(t, filepath.Join(dir, "current.json"))
	_, err := PublishTrainingBundle(dir, "old", bytes.NewReader(bundleModelFixture(0)), 0, run, bundleLimitsFixture())
	if !errors.Is(err, os.ErrExist) {
		t.Fatalf("generation collision must be exclusive: %v", err)
	}
	for _, generation := range []string{"", "../outside", "/outside", "a.b", strings.Repeat("g", 65)} {
		_, err := PublishTrainingBundle(dir, generation, bytes.NewReader(bundleModelFixture(0)), 0, run, bundleLimitsFixture())
		expectRunError(t, err, "invalid training bundle")
	}
	for _, limits := range []TrainingBundleLimits{{0, 8192}, {1024, 0}, {trainingBundleModelLimit + 1, 8192}, {1024, MaxTrainingRunStateBytes + 1}} {
		_, err := PublishTrainingBundle(dir, "invalid", bytes.NewReader(bundleModelFixture(0)), 0, run, limits)
		expectRunError(t, err, "byte ceilings")
	}
	_, err = PublishTrainingBundle(dir, "mismatch", bytes.NewReader(bundleModelFixture(1)), 1, run, bundleLimitsFixture())
	expectRunError(t, err, "step mismatch")
	raw := bundleModelFixture(0)
	binary.LittleEndian.PutUint32(raw[16:20], 2)
	_, err = PublishTrainingBundle(dir, "version", bytes.NewReader(raw), 0, run, bundleLimitsFixture())
	expectRunError(t, err, "prefix/version/step")
	_, err = PublishTrainingBundle(dir, "oversized", bytes.NewReader(append(bundleModelFixture(0), make([]byte, 1024)...)), 0, run, bundleLimitsFixture())
	expectRunError(t, err, "byte ceiling")
	if !bytes.Equal(pointer, bundleBytesFixture(t, filepath.Join(dir, "current.json"))) {
		t.Fatal("rejected publish changed pointer")
	}
	bundleReadFixture(t, dir, 0)
	// Size wrapper separately verifies actual short-write detection, not just a
	// simulated phase failure; model stream partial-read failure is below.
	w := bundleBoundedWriter{writer: bundleShortWriter{}, remaining: 10}
	if _, err := w.Write([]byte("abcd")); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("short write: %v", err)
	}
}

type bundleShortWriter struct{}

func (bundleShortWriter) Write(data []byte) (int, error) { return len(data) - 1, nil }

type bundleFailureReader struct{ prefix []byte }

func (r *bundleFailureReader) Read(dst []byte) (int, error) {
	if len(r.prefix) > 0 {
		n := copy(dst, r.prefix)
		r.prefix = r.prefix[n:]
		return n, nil
	}
	return 0, io.ErrUnexpectedEOF
}

func TestTrainingBundlePrePublicationFailures(t *testing.T) {
	for _, op := range []string{"model.write", "model.sync", "model.close", "run.write", "run.sync", "run.close", "generations.sync", "pointer.write", "pointer.sync", "pointer.close", "pointer.rename"} {
		for _, after := range []bool{false, true} {
			if op == "pointer.rename" && after {
				continue
			} // publication-visible, next test
			t.Run(op+map[bool]string{false: "-before", true: "-after"}[after], func(t *testing.T) {
				dir := t.TempDir()
				run := newRunFixture(t, 2)
				bundlePublishFixture(t, dir, "old", 0, run)
				old := bundleBytesFixture(t, filepath.Join(dir, "current.json"))
				rowAdvance(t, run, 1)
				failure := errors.New("injected-boundary-error")
				hook := func(name string, isAfter bool) error {
					if name == op && isAfter == after {
						return failure
					}
					return nil
				}
				_, err := publishTrainingBundle(dir, "new", bytes.NewReader(bundleModelFixture(1)), 1, run, bundleLimitsFixture(), hook)
				var published *TrainingBundlePublishedError
				if !errors.Is(err, failure) || errors.As(err, &published) {
					t.Fatalf("wrong failure phase: %v", err)
				}
				if !bytes.Equal(old, bundleBytesFixture(t, filepath.Join(dir, "current.json"))) {
					t.Fatal("prepublication failure altered committed pointer")
				}
				bundleReadFixture(t, dir, 0) // actual files still open old pair; ignore orphans
			})
		}
	}
	dir := t.TempDir()
	run := newRunFixture(t, 2)
	bundlePublishFixture(t, dir, "old", 0, run)
	_, err := PublishTrainingBundle(dir, "partial", &bundleFailureReader{prefix: bundleModelFixture(0)[:52]}, 0, run, bundleLimitsFixture())
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("partial stream error %v", err)
	}
	bundleReadFixture(t, dir, 0)
}
func TestTrainingBundlePublishedUncertainty(t *testing.T) {
	for _, tc := range []struct {
		op    string
		after bool
	}{{"pointer.rename", true}, {"published.sync", false}, {"published.sync", true}, {"published.close", false}, {"published.close", true}} {
		t.Run(tc.op+map[bool]string{false: "-before", true: "-after"}[tc.after], func(t *testing.T) {
			dir := t.TempDir()
			run := newRunFixture(t, 2)
			bundlePublishFixture(t, dir, "old", 0, run)
			rowAdvance(t, run, 1)
			failure := errors.New("injected-after-publication")
			_, err := publishTrainingBundle(dir, "new", bytes.NewReader(bundleModelFixture(1)), 1, run, bundleLimitsFixture(), func(op string, after bool) error {
				if op == tc.op && after == tc.after {
					return failure
				}
				return nil
			})
			var published *TrainingBundlePublishedError
			if !errors.As(err, &published) || !errors.Is(err, failure) || published.Generation != "new" {
				t.Fatalf("lost publication uncertainty: %v", err)
			}
			bundleReadFixture(t, dir, 1) // real rename succeeded, coherent NEW pair visible
			if !bytes.Equal(bundleBytesFixture(t, filepath.Join(dir, "old.model.bin")), bundleModelFixture(0)) {
				t.Fatal("uncertain publisher rolled back/deleted old")
			}
		})
	}
}
func TestTrainingBundleCorruptionAndBinding(t *testing.T) {
	for _, kind := range []string{"model-corrupt", "model-truncated", "run-corrupt", "run-truncated", "pointer-duplicate", "pointer-null", "pointer-overflow", "pointer-path", "missing-model", "missing-run", "identity"} {
		t.Run(kind, func(t *testing.T) {
			dir := t.TempDir()
			run := newRunFixture(t, 2)
			bundlePublishFixture(t, dir, "g0", 0, run)
			pointerPath := filepath.Join(dir, "current.json")
			pointer := bundleBytesFixture(t, pointerPath)
			id := runIdentityFixture()
			switch kind {
			case "model-corrupt":
				p := filepath.Join(dir, "g0.model.bin")
				raw := bundleBytesFixture(t, p)
				raw[len(raw)-1] ^= 1
				bundleWriteFixture(t, p, raw)
			case "model-truncated":
				p := filepath.Join(dir, "g0.model.bin")
				raw := bundleBytesFixture(t, p)
				bundleWriteFixture(t, p, raw[:20])
			case "run-corrupt":
				p := filepath.Join(dir, "g0.run.json")
				raw := bundleBytesFixture(t, p)
				raw[len(raw)-2] ^= 1
				bundleWriteFixture(t, p, raw)
			case "run-truncated":
				bundleWriteFixture(t, filepath.Join(dir, "g0.run.json"), []byte("{"))
			case "pointer-duplicate":
				bundleWriteFixture(t, pointerPath, bytes.Replace(pointer, []byte(`"version":1`), []byte(`"version":1,"version":1`), 1))
			case "pointer-null":
				bundleWriteFixture(t, pointerPath, bytes.Replace(pointer, []byte(`"model_bytes":65`), []byte(`"model_bytes":null`), 1))
			case "pointer-overflow":
				var p map[string]json.RawMessage
				if err := json.Unmarshal(pointer, &p); err != nil {
					t.Fatal(err)
				}
				p["model_bytes"] = json.RawMessage("18446744073709551616")
				raw, err := json.Marshal(p)
				if err != nil {
					t.Fatal(err)
				}
				bundleWriteFixture(t, pointerPath, raw)
			case "pointer-path":
				bundleWriteFixture(t, pointerPath, bytes.Replace(pointer, []byte(`"generation":"g0"`), []byte(`"generation":"../g0"`), 1))
			case "missing-model":
				if err := os.Remove(filepath.Join(dir, "g0.model.bin")); err != nil {
					t.Fatal(err)
				}
			case "missing-run":
				if err := os.Remove(filepath.Join(dir, "g0.run.json")); err != nil {
					t.Fatal(err)
				}
			case "identity":
				id.GOARCH = "arm64"
			}
			pair, err := OpenTrainingBundle(dir, id, bundleLimitsFixture())
			if err == nil || pair != nil {
				if pair != nil {
					pair.Close()
				}
				t.Fatalf("corrupt %s exposed pair", kind)
			}
		})
	}
	dir := t.TempDir()
	run := newRunFixture(t, 1)
	bundlePublishFixture(t, dir, "g0", 0, run)
	// No committed pointer means even complete generation files are not selected.
	if err := os.Remove(filepath.Join(dir, "current.json")); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenTrainingBundle(dir, runIdentityFixture(), bundleLimitsFixture()); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("orphan selected: %v", err)
	}
	limits := bundleLimitsFixture()
	p := trainingBundlePointer{Version: 1, Generation: "g0", ModelName: "g0.model.bin", ModelSHA256: strings.Repeat("8", 64), ModelBytes: 65, RunName: "g0.run.json", RunSHA256: strings.Repeat("9", 64), RunBytes: 10, Step: uint64(math.MaxInt) + 1}
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	_, err = decodeBundlePointer(data, limits)
	expectRunError(t, err, "step bound")
}
func TestTrainingBundleSpecialFiles(t *testing.T) {
	for _, name := range []string{"current.json", "g0.model.bin", "g0.run.json"} {
		for _, kind := range []string{"symlink", "fifo"} {
			t.Run(name+"-"+kind, func(t *testing.T) {
				dir := t.TempDir()
				run := newRunFixture(t, 1)
				bundlePublishFixture(t, dir, "g0", 0, run)
				target := filepath.Join(dir, name)
				if err := os.Remove(target); err != nil {
					t.Fatal(err)
				}
				if kind == "symlink" {
					outside := filepath.Join(t.TempDir(), "target")
					bundleWriteFixture(t, outside, []byte("unchanged"))
					if err := os.Symlink(outside, target); err != nil {
						t.Fatal(err)
					}
				} else {
					if err := syscall.Mkfifo(target, 0600); err != nil {
						t.Fatal(err)
					}
				}
				pair, err := OpenTrainingBundle(dir, runIdentityFixture(), bundleLimitsFixture())
				if err == nil || pair != nil {
					t.Fatal("link/special input admitted")
				}
				expectRunError(t, err, "regular non-symlink")
				if name == "current.json" {
					_, err := PublishTrainingBundle(dir, "new", bytes.NewReader(bundleModelFixture(0)), 0, run, bundleLimitsFixture())
					expectRunError(t, err, "existing bundle pointer")
				}
			})
		}
	}
	dir := t.TempDir()
	outside := filepath.Join(t.TempDir(), "outside")
	bundleWriteFixture(t, outside, []byte("unchanged"))
	if err := os.Symlink(outside, filepath.Join(dir, "g0.model.bin")); err != nil {
		t.Fatal(err)
	}
	_, err := PublishTrainingBundle(dir, "g0", bytes.NewReader(bundleModelFixture(0)), 0, newRunFixture(t, 1), bundleLimitsFixture())
	if !errors.Is(err, os.ErrExist) {
		t.Fatalf("generation link followed: %v", err)
	}
	if string(bundleBytesFixture(t, outside)) != "unchanged" {
		t.Fatal("outside symlink target altered")
	}
}
