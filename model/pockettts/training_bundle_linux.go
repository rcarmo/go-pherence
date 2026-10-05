//go:build linux

package pockettts

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"syscall"
)

const trainingBundlePointerLimit = 4096
const trainingBundleModelLimit int64 = 16 << 30
const trainingBundlePrefixBytes = 52 // magic[16], version[4], step[8], optimiser fields[24]

// TrainingBundleLimits bounds bytes, not runtime/RSS/disk availability. The
// caller supplies a trusted finite Reader: a byte ceiling cannot interrupt a
// Reader blocked in Read. Publication requires one exclusive directory writer
// and a coherent immutable model stream/run state; neither is locked here.
type TrainingBundleLimits struct {
	MaxModelBytes   int64
	MaxSidecarBytes int64
}

// TrainingBundlePublishedError means current.json was replaced successfully.
// The coherent pair is visible but subsequent sync/close could not establish
// durability. Do not rollback, delete generations or retry automatically.
type TrainingBundlePublishedError struct {
	Generation string
	Err        error
}

func (e *TrainingBundlePublishedError) Error() string {
	return fmt.Sprintf("training bundle %s is publication-visible, durability uncertain: %v", e.Generation, e.Err)
}
func (e *TrainingBundlePublishedError) Unwrap() error { return e.Err }

type trainingBundlePointer struct {
	Version     uint64 `json:"version"`
	Generation  string `json:"generation"`
	ModelName   string `json:"model_name"`
	ModelSHA256 string `json:"model_sha256"`
	ModelBytes  int64  `json:"model_bytes"`
	RunName     string `json:"run_name"`
	RunSHA256   string `json:"run_sha256"`
	RunBytes    int64  `json:"run_bytes"`
	Step        uint64 `json:"step"`
}

// PublishTrainingBundle publishes association only. It checks binary prefix
// version/step, NOT tensor/trailer/optimiser validity. The caller serializes and
// validates the model separately. New immutable generation files are exclusive;
// errors may leave unselected orphans, which must not be reused or auto-loaded.
func PublishTrainingBundle(directory, generation string, model io.Reader, modelStep uint64, run *TrainingRunState, limits TrainingBundleLimits) (TrainingCheckpointBinding, error) {
	return publishTrainingBundle(directory, generation, model, modelStep, run, limits, nil)
}

// Tiny operation seam for before/after error injection. Production is nil;
// no global hooks, filesystem framework or automatic retry are introduced.
type trainingBundleHook func(operation string, after bool) error

func bundleOperation(h trainingBundleHook, name string, operation func() error) error {
	if h != nil {
		if err := h(name, false); err != nil {
			return fmt.Errorf("before %s: %w", name, err)
		}
	}
	if err := operation(); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	if h != nil {
		if err := h(name, true); err != nil {
			return fmt.Errorf("after %s: %w", name, err)
		}
	}
	return nil
}

func publishTrainingBundle(directory, generation string, model io.Reader, modelStep uint64, run *TrainingRunState, limits TrainingBundleLimits, h trainingBundleHook) (binding TrainingCheckpointBinding, err error) {
	if err = limits.validate(); err != nil {
		return binding, err
	}
	if !bundleGeneration(generation) || model == nil || run == nil || modelStep > uint64(math.MaxInt) {
		return binding, fmt.Errorf("invalid training bundle generation/model/step")
	}
	// Deep-copy all run storage before directory access. Single-owner contract
	// still forbids concurrent mutation while taking this snapshot.
	snapshot := &TrainingRunState{core: run.core.clone()}
	if err = snapshot.core.validate(); err != nil {
		return binding, err
	}
	if snapshot.core.step != modelStep {
		return binding, fmt.Errorf("bundle model/run step mismatch")
	}
	binding = TrainingCheckpointBinding{Name: generation + ".model.bin", SHA256: string(bytes.Repeat([]byte{'0'}, 64)), Step: modelStep}
	// Validate identity, file naming and eventual sidecar bound before writes.
	probe, e := snapshot.Snapshot(binding)
	if e != nil {
		return binding, e
	}
	if int64(len(probe)) > limits.MaxSidecarBytes {
		return binding, fmt.Errorf("bundle sidecar byte ceiling")
	}
	var prefix [trainingBundlePrefixBytes]byte
	if _, err = io.ReadFull(model, prefix[:]); err != nil {
		return binding, err
	}
	if err = checkBundleModelPrefix(prefix[:], modelStep); err != nil {
		return binding, err
	}
	root, e := os.OpenRoot(directory)
	if e != nil {
		return binding, e
	}
	var directoryFile *os.File
	published := false
	defer func() {
		if directoryFile != nil {
			err = errors.Join(err, directoryFile.Close())
		}
		err = errors.Join(err, root.Close())
		if published && err != nil {
			err = &TrainingBundlePublishedError{Generation: generation, Err: err}
		}
	}()
	directoryFile, err = root.OpenFile(".", os.O_RDONLY|syscall.O_NOFOLLOW|syscall.O_NONBLOCK, 0)
	if err != nil {
		return binding, err
	}
	if err = bundlePointerAdmission(root); err != nil {
		return binding, err
	}
	modelFile, e := root.OpenFile(binding.Name, os.O_CREATE|os.O_EXCL|os.O_WRONLY|syscall.O_NOFOLLOW|syscall.O_NONBLOCK, 0600)
	if e != nil {
		return binding, e
	}
	modelClosed := false
	defer func() {
		if !modelClosed {
			err = errors.Join(err, modelFile.Close())
		}
	}()
	modelDigest := sha256.New()
	writer := &bundleBoundedWriter{writer: io.MultiWriter(modelFile, modelDigest), remaining: limits.MaxModelBytes}
	err = bundleOperation(h, "model.write", func() error {
		if _, e := writer.Write(prefix[:]); e != nil {
			return e
		}
		_, e := io.CopyBuffer(writer, io.LimitReader(model, limits.MaxModelBytes-trainingBundlePrefixBytes+1), make([]byte, 32<<10))
		return e
	})
	if err != nil {
		return binding, err
	}
	if err = bundleOperation(h, "model.sync", modelFile.Sync); err != nil {
		return binding, err
	}
	if err = bundleOperation(h, "model.close", func() error { modelClosed = true; return modelFile.Close() }); err != nil {
		return binding, err
	}
	binding.SHA256 = hex.EncodeToString(modelDigest.Sum(nil))
	data, e := snapshot.Snapshot(binding)
	if e != nil {
		return binding, e
	}
	runName := generation + ".run.json"
	if err = writeBundleFile(root, runName, data, limits.MaxSidecarBytes, "run", h); err != nil {
		return binding, err
	}
	if err = bundleOperation(h, "generations.sync", directoryFile.Sync); err != nil {
		return binding, err
	}
	runDigest := sha256.Sum256(data)
	pointer := trainingBundlePointer{Version: 1, Generation: generation, ModelName: binding.Name, ModelSHA256: binding.SHA256, ModelBytes: writer.written, RunName: runName, RunSHA256: hex.EncodeToString(runDigest[:]), RunBytes: int64(len(data)), Step: modelStep}
	pointerBytes, e := json.Marshal(pointer)
	if e != nil {
		return binding, e
	}
	temp := generation + ".pointer.tmp"
	if err = writeBundleFile(root, temp, pointerBytes, trainingBundlePointerLimit, "pointer", h); err != nil {
		return binding, err
	}
	err = bundleOperation(h, "pointer.rename", func() error {
		if e := root.Rename(temp, "current.json"); e != nil {
			return e
		}
		published = true // before any after-operation hook or subsequent close
		return nil
	})
	if err != nil {
		return binding, err
	}
	if err = bundleOperation(h, "published.sync", directoryFile.Sync); err != nil {
		return binding, err
	}
	err = bundleOperation(h, "published.close", func() error { f := directoryFile; directoryFile = nil; return f.Close() })
	return binding, err
}

func writeBundleFile(root *os.Root, name string, data []byte, limit int64, phase string, h trainingBundleHook) (err error) {
	if len(data) == 0 || int64(len(data)) > limit {
		return fmt.Errorf("%s byte ceiling", phase)
	}
	f, e := root.OpenFile(name, os.O_CREATE|os.O_EXCL|os.O_WRONLY|syscall.O_NOFOLLOW|syscall.O_NONBLOCK, 0600)
	if e != nil {
		return e
	}
	closed := false
	defer func() {
		if !closed {
			err = errors.Join(err, f.Close())
		}
	}()
	if err = bundleOperation(h, phase+".write", func() error {
		n, e := f.Write(data)
		if e == nil && n != len(data) {
			return io.ErrShortWrite
		}
		return e
	}); err != nil {
		return err
	}
	if err = bundleOperation(h, phase+".sync", f.Sync); err != nil {
		return err
	}
	return bundleOperation(h, phase+".close", func() error { closed = true; return f.Close() })
}

type bundleBoundedWriter struct {
	writer             io.Writer
	remaining, written int64
}

func (w *bundleBoundedWriter) Write(data []byte) (int, error) {
	if int64(len(data)) > w.remaining {
		return 0, fmt.Errorf("bundle model byte ceiling exceeded")
	}
	n, err := w.writer.Write(data)
	w.remaining -= int64(n)
	w.written += int64(n)
	if err == nil && n != len(data) {
		err = io.ErrShortWrite
	}
	return n, err
}
func (l TrainingBundleLimits) validate() error {
	if l.MaxModelBytes < trainingBundlePrefixBytes || l.MaxModelBytes > trainingBundleModelLimit || l.MaxSidecarBytes <= 0 || l.MaxSidecarBytes > MaxTrainingRunStateBytes {
		return fmt.Errorf("invalid training bundle byte ceilings")
	}
	return nil
}
func bundleGeneration(name string) bool {
	if len(name) == 0 || len(name) > 64 {
		return false
	}
	for _, c := range name {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_') {
			return false
		}
	}
	return true
}
func checkBundleModelPrefix(prefix []byte, step uint64) error {
	if len(prefix) < trainingBundlePrefixBytes || !bytes.Equal(prefix[:16], fullTrainingBinaryMagic[:]) || binary.LittleEndian.Uint32(prefix[16:20]) != 1 || binary.LittleEndian.Uint64(prefix[20:28]) != step || step > uint64(math.MaxInt) {
		return fmt.Errorf("bundle model prefix/version/step mismatch")
	}
	return nil
}

// TrainingBundleRead owns the same descriptor whose bytes were checked. Close
// it after separate bounded model decoding/validation and before live mutation.
// Association checking does not validate the binary checksum/tensor contents.
type TrainingBundleRead struct {
	Model      *os.File
	Run        *TrainingRunState
	Checkpoint TrainingCheckpointBinding
	ModelBytes int64
}

func (b *TrainingBundleRead) Close() error {
	if b == nil || b.Model == nil {
		return nil
	}
	f := b.Model
	b.Model = nil
	return f.Close()
}

// OpenTrainingBundle follows only current.json, never directory-scanned orphan
// generations. It rejects known links/special files and verifies size/digest/
// prefix/step association without tensor decoding. Single-writer immutable
// generations are a precondition; adversarial concurrent modification and
// actual power-loss/crash durability are not certified by this API.
func OpenTrainingBundle(directory string, expected TrainingRunIdentity, limits TrainingBundleLimits) (result *TrainingBundleRead, err error) {
	if err = limits.validate(); err != nil {
		return nil, err
	}
	if err = expected.validate(); err != nil {
		return nil, err
	}
	root, e := os.OpenRoot(directory)
	if e != nil {
		return nil, e
	}
	defer func() {
		if e := root.Close(); e != nil {
			err = errors.Join(err, e)
			if result != nil {
				err = errors.Join(err, result.Close())
				result = nil
			}
		}
	}()
	pointerFile, e := openBundleRegular(root, "current.json", trainingBundlePointerLimit)
	if e != nil {
		return nil, e
	}
	pointerData, e := io.ReadAll(io.LimitReader(pointerFile, trainingBundlePointerLimit+1))
	ce := pointerFile.Close()
	if e != nil || ce != nil {
		return nil, errors.Join(e, ce)
	}
	pointer, e := decodeBundlePointer(pointerData, limits)
	if e != nil {
		return nil, e
	}
	model, e := openBundleRegular(root, pointer.ModelName, limits.MaxModelBytes)
	if e != nil {
		return nil, e
	}
	defer func() {
		if result == nil {
			err = errors.Join(err, model.Close())
		}
	}()
	st, e := model.Stat()
	if e != nil {
		return nil, e
	}
	if st.Size() != pointer.ModelBytes {
		return nil, fmt.Errorf("bundle model byte count mismatch")
	}
	var prefix [trainingBundlePrefixBytes]byte
	if _, e = io.ReadFull(model, prefix[:]); e != nil {
		return nil, e
	}
	if e = checkBundleModelPrefix(prefix[:], pointer.Step); e != nil {
		return nil, e
	}
	if _, e = model.Seek(0, io.SeekStart); e != nil {
		return nil, e
	}
	h := sha256.New()
	n, e := io.CopyBuffer(h, io.LimitReader(model, pointer.ModelBytes+1), make([]byte, 32<<10))
	if e != nil {
		return nil, e
	}
	if n != pointer.ModelBytes || hex.EncodeToString(h.Sum(nil)) != pointer.ModelSHA256 {
		return nil, fmt.Errorf("bundle model digest/size mismatch")
	}
	runFile, e := openBundleRegular(root, pointer.RunName, limits.MaxSidecarBytes)
	if e != nil {
		return nil, e
	}
	st, e = runFile.Stat()
	if e != nil || st.Size() != pointer.RunBytes {
		ce := runFile.Close()
		return nil, errors.Join(fmt.Errorf("bundle sidecar byte count mismatch"), e, ce)
	}
	data, e := io.ReadAll(io.LimitReader(runFile, pointer.RunBytes+1))
	ce = runFile.Close()
	if e != nil || ce != nil {
		return nil, errors.Join(e, ce)
	}
	digest := sha256.Sum256(data)
	if int64(len(data)) != pointer.RunBytes || hex.EncodeToString(digest[:]) != pointer.RunSHA256 {
		return nil, fmt.Errorf("bundle sidecar digest/size mismatch")
	}
	binding := TrainingCheckpointBinding{Name: pointer.ModelName, SHA256: pointer.ModelSHA256, Step: pointer.Step}
	run, e := RestoreTrainingRunState(data, expected, binding)
	if e != nil {
		return nil, e
	}
	if _, e = model.Seek(0, io.SeekStart); e != nil {
		return nil, e
	}
	return &TrainingBundleRead{Model: model, Run: run, Checkpoint: binding, ModelBytes: pointer.ModelBytes}, nil
}
func openBundleRegular(root *os.Root, name string, limit int64) (*os.File, error) {
	st, e := root.Lstat(name)
	if e != nil {
		return nil, e
	}
	if !st.Mode().IsRegular() {
		return nil, fmt.Errorf("bundle input %s is not a regular non-symlink file", name)
	}
	f, e := root.OpenFile(name, os.O_RDONLY|syscall.O_NOFOLLOW|syscall.O_NONBLOCK, 0)
	if e != nil {
		return nil, e
	}
	st, e = f.Stat()
	if e != nil {
		ce := f.Close()
		return nil, errors.Join(e, ce)
	}
	if !st.Mode().IsRegular() || st.Size() <= 0 || st.Size() > limit {
		ce := f.Close()
		return nil, errors.Join(fmt.Errorf("bundle input %s exceeds regular-file byte bound", name), ce)
	}
	return f, nil
}
func bundlePointerAdmission(root *os.Root) error {
	st, e := root.Lstat("current.json")
	if errors.Is(e, os.ErrNotExist) {
		return nil
	}
	if e != nil {
		return e
	}
	if !st.Mode().IsRegular() {
		return fmt.Errorf("existing bundle pointer is not a regular non-symlink file")
	}
	return nil
}
func decodeBundlePointer(data []byte, limits TrainingBundleLimits) (trainingBundlePointer, error) {
	var p trainingBundlePointer
	if len(data) == 0 || len(data) > trainingBundlePointerLimit {
		return p, fmt.Errorf("bundle pointer byte bound")
	}
	fields, e := runJSONObject(data, []string{"version", "generation", "model_name", "model_sha256", "model_bytes", "run_name", "run_sha256", "run_bytes", "step"})
	if e != nil {
		return p, e
	}
	p.Version, e = runUint(fields["version"])
	if e != nil || p.Version != 1 {
		return p, fmt.Errorf("bundle pointer schema")
	}
	p.Generation, e = runString(fields["generation"], 64)
	if e != nil || !bundleGeneration(p.Generation) {
		return p, fmt.Errorf("bundle generation name")
	}
	for _, v := range []struct {
		key string
		dst *string
	}{{"model_name", &p.ModelName}, {"run_name", &p.RunName}, {"model_sha256", &p.ModelSHA256}, {"run_sha256", &p.RunSHA256}} {
		s, e := runString(fields[v.key], 128)
		if e != nil {
			return p, e
		}
		*v.dst = s
	}
	if p.ModelName != p.Generation+".model.bin" || p.RunName != p.Generation+".run.json" || !runHex(p.ModelSHA256, 64) || !runHex(p.RunSHA256, 64) {
		return p, fmt.Errorf("bundle pointer names/digests")
	}
	n, e := runUint(fields["model_bytes"])
	if e != nil || n < trainingBundlePrefixBytes || n > uint64(limits.MaxModelBytes) {
		return p, fmt.Errorf("bundle pointer model byte count")
	}
	p.ModelBytes = int64(n)
	n, e = runUint(fields["run_bytes"])
	if e != nil || n == 0 || n > uint64(limits.MaxSidecarBytes) {
		return p, fmt.Errorf("bundle pointer sidecar byte count")
	}
	p.RunBytes = int64(n)
	p.Step, e = runUint(fields["step"])
	if e != nil || p.Step > uint64(math.MaxInt) {
		return p, fmt.Errorf("bundle pointer step bound")
	}
	return p, nil
}
