package expertstream

import (
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"
)

type Reader struct {
	mu        sync.Mutex
	closed    bool
	manifest  Manifest
	alignment int64
	slotSize  int64
	workers   int
	files     map[string]*fileRef
	experts   map[uint64]*expertLayout
	slots     []*slot
	keyToSlot map[uint64]int
	clock     uint64
}

// Open validates a manifest package, verifies file checksums, opens the
// referenced files, and allocates a fixed number of aligned host slots.
func Open(manifestPath string, opts Options) (*Reader, error) {
	if opts.Slots <= 0 {
		return nil, fmt.Errorf("expertstream: slots must be positive")
	}
	manifest, files, experts, maxSpan, err := openManifest(manifestPath)
	if err != nil {
		return nil, err
	}
	workers := opts.Workers
	if workers <= 0 {
		workers = 1
	}
	slotSize, err := alignUp(maxSpan, manifest.Alignment)
	if err != nil {
		closeFiles(files)
		return nil, err
	}
	slots := make([]*slot, opts.Slots)
	for i := range slots {
		raw, buf, err := allocAligned(slotSize, manifest.Alignment)
		if err != nil {
			for j := 0; j < i; j++ {
				_ = freeAligned(slots[j].raw)
			}
			closeFiles(files)
			return nil, err
		}
		slots[i] = &slot{index: i, raw: raw, buf: buf}
	}
	return &Reader{
		manifest:  cloneManifest(manifest),
		alignment: manifest.Alignment,
		slotSize:  slotSize,
		workers:   workers,
		files:     files,
		experts:   experts,
		slots:     slots,
		keyToSlot: make(map[uint64]int, len(slots)),
	}, nil
}

// Manifest returns a defensive copy of the validated manifest.
func (r *Reader) Manifest() Manifest {
	r.mu.Lock()
	defer r.mu.Unlock()
	return cloneManifest(r.manifest)
}

// Close releases slot memory and closes open data files.
func (r *Reader) Close() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true
	var errs []error
	for _, sl := range r.slots {
		if sl == nil {
			continue
		}
		if err := freeAligned(sl.raw); err != nil {
			errs = append(errs, err)
		}
		sl.raw = nil
		sl.buf = nil
	}
	for _, f := range r.files {
		if f != nil && f.file != nil {
			if err := f.file.Close(); err != nil {
				errs = append(errs, err)
			}
			f.file = nil
		}
	}
	if len(errs) == 0 {
		return nil
	}
	return errors.Join(errs...)
}

// Load resolves the requested keys in caller order. Each result references a
// reusable aligned slot owned by the Reader; returned bytes must be treated as
// read-only and remain valid only until the slot is reused by a later Load.
func (r *Reader) Load(keys []uint64) ([]LoadedExpert, error) {
	if len(keys) == 0 {
		return nil, nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, ErrClosed
	}

	unique := make(map[uint64]struct{}, len(keys))
	missing := make([]uint64, 0, len(keys))
	keepSlots := make(map[int]struct{}, len(keys))
	for _, key := range keys {
		if _, ok := unique[key]; ok {
			continue
		}
		unique[key] = struct{}{}
		if _, ok := r.experts[key]; !ok {
			return nil, fmt.Errorf("%w: key=%d", ErrUnknownExpert, key)
		}
		if slotIndex, ok := r.keyToSlot[key]; ok {
			keepSlots[slotIndex] = struct{}{}
			r.touchSlotLocked(slotIndex)
		} else {
			missing = append(missing, key)
		}
	}
	if len(unique) > len(r.slots) {
		return nil, fmt.Errorf("%w: need=%d slots=%d", ErrSlotCapacity, len(unique), len(r.slots))
	}

	assignments, err := r.assignSlotsLocked(missing, keepSlots)
	if err != nil {
		return nil, err
	}
	if err := r.loadAssignmentsLocked(assignments); err != nil {
		r.discardAssignmentsLocked(assignments)
		return nil, err
	}
	for _, assignment := range assignments {
		sl := r.slots[assignment.slotIndex]
		if sl.loaded {
			delete(r.keyToSlot, sl.key)
		}
		sl.loaded = true
		sl.key = assignment.key
		sl.span = assignment.layout.span
		r.touchSlotLocked(assignment.slotIndex)
		r.keyToSlot[assignment.key] = assignment.slotIndex
	}

	out := make([]LoadedExpert, len(keys))
	for i, key := range keys {
		layout := r.experts[key]
		slotIndex := r.keyToSlot[key]
		sl := r.slots[slotIndex]
		span := sl.buf[:layout.span]
		upStart := int(layout.upRel)
		downStart := int(layout.downRel)
		upEnd := upStart + int(layout.spec.Up.Size)
		downEnd := downStart + int(layout.spec.Down.Size)
		out[i] = LoadedExpert{
			Key: key,
			Slot: SlotView{
				Index: slotIndex,
				Bytes: span,
			},
			Gate: Component{
				DType: layout.spec.Gate.DType,
				Shape: append([]int64(nil), layout.spec.Gate.Shape...),
				Bytes: span[:layout.spec.Gate.Size],
			},
			Up: Component{
				DType: layout.spec.Up.DType,
				Shape: append([]int64(nil), layout.spec.Up.Shape...),
				Bytes: span[upStart:upEnd],
			},
			Down: Component{
				DType: layout.spec.Down.DType,
				Shape: append([]int64(nil), layout.spec.Down.Shape...),
				Bytes: span[downStart:downEnd],
			},
		}
	}
	return out, nil
}

type slotAssignment struct {
	key       uint64
	layout    *expertLayout
	slotIndex int
}

func (r *Reader) assignSlotsLocked(missing []uint64, keepSlots map[int]struct{}) ([]slotAssignment, error) {
	if len(missing) == 0 {
		return nil, nil
	}
	unloaded := make([]int, 0, len(r.slots))
	victims := make([]int, 0, len(r.slots))
	for i, sl := range r.slots {
		if _, keep := keepSlots[i]; keep {
			continue
		}
		if sl.loaded {
			victims = append(victims, i)
		} else {
			unloaded = append(unloaded, i)
		}
	}
	sort.Ints(unloaded)
	sort.Slice(victims, func(i, j int) bool {
		a := r.slots[victims[i]]
		b := r.slots[victims[j]]
		if a.lastUse != b.lastUse {
			return a.lastUse < b.lastUse
		}
		return a.index < b.index
	})
	candidates := append(unloaded, victims...)
	if len(candidates) < len(missing) {
		return nil, fmt.Errorf("%w: need=%d available=%d", ErrSlotCapacity, len(missing), len(candidates))
	}
	assignments := make([]slotAssignment, len(missing))
	for i, key := range missing {
		assignments[i] = slotAssignment{key: key, layout: r.experts[key], slotIndex: candidates[i]}
	}
	return assignments, nil
}

func (r *Reader) loadAssignmentsLocked(assignments []slotAssignment) error {
	if len(assignments) == 0 {
		return nil
	}
	workerCount := r.workers
	if workerCount <= 0 {
		workerCount = 1
	}
	if workerCount > len(assignments) {
		workerCount = len(assignments)
	}
	jobs := make(chan slotAssignment)
	errCh := make(chan error, len(assignments))
	var wg sync.WaitGroup
	for i := 0; i < workerCount; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for assignment := range jobs {
				dst := r.slots[assignment.slotIndex].buf[:assignment.layout.span]
				if err := readFullAt(assignment.layout.file.file, assignment.layout.file.absPath, dst, assignment.layout.spec.Gate.Offset); err != nil {
					errCh <- fmt.Errorf("expert key=%d: %w", assignment.key, err)
				}
			}
		}()
	}
	for _, assignment := range assignments {
		jobs <- assignment
	}
	close(jobs)
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			return err
		}
	}
	return nil
}

func (r *Reader) discardAssignmentsLocked(assignments []slotAssignment) {
	for _, assignment := range assignments {
		sl := r.slots[assignment.slotIndex]
		if sl.loaded {
			delete(r.keyToSlot, sl.key)
		}
		sl.loaded = false
		sl.key = 0
		sl.span = 0
	}
}

func (r *Reader) touchSlotLocked(index int) {
	r.clock++
	r.slots[index].lastUse = r.clock
}

func readFullAt(f interface {
	ReadAt([]byte, int64) (int, error)
}, path string, dst []byte, offset int64) error {
	if len(dst) == 0 {
		return nil
	}
	done := 0
	for done < len(dst) {
		n, err := f.ReadAt(dst[done:], offset+int64(done))
		if n > 0 {
			done += n
		}
		if err != nil {
			// ReaderAt implementations may legally return the final bytes and
			// io.EOF together. The read is complete in that case.
			if done == len(dst) {
				return nil
			}
			if errors.Is(err, io.EOF) {
				return fmt.Errorf("%w: file=%s offset=%d read=%d want=%d", ErrShortRead, path, offset, done, len(dst))
			}
			return fmt.Errorf("file=%s offset=%d: %w", path, offset+int64(done), err)
		}
		if n == 0 {
			return fmt.Errorf("%w: file=%s offset=%d read=%d want=%d", ErrShortRead, path, offset, done, len(dst))
		}
	}
	return nil
}

func normalizeDType(s string) string {
	return strings.ToLower(strings.TrimSpace(s))
}

func dtypeBytes(dtype string) (int, bool) {
	switch normalizeDType(dtype) {
	case "u8", "i8":
		return 1, true
	case "u16", "i16", "f16", "bf16":
		return 2, true
	case "u32", "i32", "f32":
		return 4, true
	case "u64", "i64", "f64":
		return 8, true
	default:
		return 0, false
	}
}

func isPowerOfTwo(v int64) bool {
	return v > 0 && (v&(v-1)) == 0
}

func alignUp(v, alignment int64) (int64, error) {
	if !isPowerOfTwo(alignment) {
		return 0, fmt.Errorf("alignment=%d must be a positive power of two", alignment)
	}
	if v < 0 {
		return 0, fmt.Errorf("value must be non-negative")
	}
	mask := alignment - 1
	if v > 0 && v > ((1<<63)-1)-mask {
		return 0, fmt.Errorf("align overflow")
	}
	return (v + mask) &^ mask, nil
}

func checkedAdd(a, b int64) (int64, error) {
	if b > 0 && a > ((1<<63)-1)-b {
		return 0, fmt.Errorf("overflow")
	}
	if b < 0 && a < (-1<<63)-b {
		return 0, fmt.Errorf("overflow")
	}
	return a + b, nil
}

func checkedProduct(a, b int64) (int64, error) {
	if a == 0 || b == 0 {
		return 0, nil
	}
	if a > 0 && b > 0 && a > ((1<<63)-1)/b {
		return 0, fmt.Errorf("overflow")
	}
	if a < 0 && b < 0 && a < ((1<<63)-1)/b {
		return 0, fmt.Errorf("overflow")
	}
	if a > 0 && b < 0 && b < (-1<<63)/a {
		return 0, fmt.Errorf("overflow")
	}
	if a < 0 && b > 0 && a < (-1<<63)/b {
		return 0, fmt.Errorf("overflow")
	}
	return a * b, nil
}
