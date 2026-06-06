// Package npu is a pure-Go userspace driver for the SpaceMIT K3 NPU's
// tightly-coupled memory (TCM) and DMA rings, reverse-engineered from
// libspine_tcm.so + the SpaceMIT ONNX Runtime EP. No cgo.
//
// Device model (see research/npu-whisper/NPU_ABI.md):
//   /dev/tcm        on-chip SRAM scratchpad, O_RDWR|O_SYNC, ioctl+mmap
//   /dev/ai_dma     DMA engine (MMIO doorbell)
//   /dev/aidma_list DMA descriptor ring (mmap, 4 KiB)
//   /dev/dma_msi    MSI completion page (mmap, 4 KiB)
//
// TCM is 8 cores x 384 KiB. Acquisition is a kernel ioctl; the EP's
// /dev/shm/tcm_sync_standalone lock is userspace-only cross-process
// bookkeeping, so a process that owns the NPU exclusively is immune to the
// "tcm buffer acquire failed" leak.
//
// Go quirk: the driver's per-core mmap sequence must not be interleaved with
// other mmap syscalls. Go's runtime (GC/heap growth) races it, so Open
// disables GC + locks the OS thread + reserves a contiguous range and
// MAP_FIXED-maps each core into it with a retry loop. Cores 6-7 may not
// acquire under this race; their memory is still usable via the reserve.
package npu

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"unsafe"

	"golang.org/x/sys/unix"
)

const (
	tcmInfoGet     = 0x80046307 // _IOR('c',7,4): info struct, block size at +8
	tcmAcquire     = 0xc0046309 // _IOWR('c',9,4): arg = core id 0..7
	defaultBlock   = 393216     // 384 KiB per core
	defaultCores   = 8
	ringBytes      = 4096
	acquireRetries = 4096
)

// TCM holds the open NPU device handles and mmap'd regions.
type TCM struct {
	tcmFd, aiFd, listFd, msiFd int
	BlockSize                  int      // per-core TCM bytes (393216)
	NumCores                   int      // total cores (8)
	Acquired                   []int    // cores successfully acquired for compute
	Mem                        []byte   // contiguous TCM mmap (NumCores*BlockSize)
	Cores                      [][]byte // per-core sub-slices of Mem
	Ring                       []byte   // aidma_list descriptor ring (4 KiB)
	MSI                        []byte   // dma_msi completion page (4 KiB)
	base                       uintptr
}

func ioctl(fd int, req uintptr, arg unsafe.Pointer) unix.Errno {
	_, _, e := unix.Syscall(unix.SYS_IOCTL, uintptr(fd), req, uintptr(arg))
	return e
}

// Available reports whether the NPU TCM device can be opened.
func Available() bool {
	fd, err := unix.Open("/dev/tcm", unix.O_RDWR|unix.O_SYNC, 0)
	if err != nil {
		return false
	}
	unix.Close(fd)
	return true
}

// Open opens the NPU devices, queries TCM geometry, maps a contiguous TCM
// region, acquires as many cores as the driver allows, and maps the DMA rings.
// Call Close to release.
func Open() (*TCM, error) {
	t := &TCM{tcmFd: -1, aiFd: -1, listFd: -1, msiFd: -1, NumCores: defaultCores, BlockSize: defaultBlock}

	fd, err := unix.Open("/dev/tcm", unix.O_RDWR|unix.O_SYNC, 0)
	if err != nil {
		return nil, fmt.Errorf("open /dev/tcm: %w", err)
	}
	t.tcmFd = fd

	var info [16]uint32
	if e := ioctl(fd, tcmInfoGet, unsafe.Pointer(&info)); e != 0 {
		t.Close()
		return nil, fmt.Errorf("TCM_INFO_GET: %w", e)
	}
	if info[2] != 0 {
		t.BlockSize = int(info[2])
	}

	// The per-core mmap+acquire sequence must run without intervening mmaps.
	// Disable GC and pin the thread for the duration.
	prev := debug.SetGCPercent(-1)
	runtime.LockOSThread()
	defer func() {
		runtime.UnlockOSThread()
		debug.SetGCPercent(prev)
	}()

	bs := uintptr(t.BlockSize)
	total := bs * uintptr(t.NumCores)
	ptr, err := unix.MmapPtr(fd, 0, nil, total, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		t.Close()
		return nil, fmt.Errorf("mmap tcm reserve: %w", err)
	}
	base := uintptr(ptr)
	t.base = base
	t.Mem = unsafe.Slice((*byte)(ptr), int(total))
	t.Cores = make([][]byte, t.NumCores)
	for c := 0; c < t.NumCores; c++ {
		t.Cores[c] = t.Mem[c*t.BlockSize : (c+1)*t.BlockSize]
	}

	// Per-core: MAP_FIXED into the reserve, then acquire. Retry cores that a
	// racing runtime mmap interrupted.
	done := make([]bool, t.NumCores)
	var core [defaultCores]uint32
	n := 0
	for attempt := 0; attempt < acquireRetries && n < t.NumCores; attempt++ {
		for c := 0; c < t.NumCores; c++ {
			if done[c] {
				continue
			}
			want := base + uintptr(c)*bs
			a, _, _ := unix.Syscall6(unix.SYS_MMAP, want, bs,
				unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED|unix.MAP_FIXED, uintptr(fd), uintptr(c)*bs)
			if a != want {
				continue
			}
			core[c] = uint32(c)
			if ie := ioctl(fd, tcmAcquire, unsafe.Pointer(&core[c])); ie == 0 {
				done[c] = true
				n++
			}
		}
	}
	for c := 0; c < t.NumCores; c++ {
		if done[c] {
			t.Acquired = append(t.Acquired, c)
		}
	}
	if n == 0 {
		t.Close()
		return nil, fmt.Errorf("acquired no TCM cores")
	}

	// DMA rings (needed for GEMM, best-effort here).
	if fd, err := unix.Open("/dev/ai_dma", unix.O_RDWR, 0); err == nil {
		t.aiFd = fd
	}
	if fd, err := unix.Open("/dev/aidma_list", unix.O_RDWR, 0); err == nil {
		t.listFd = fd
		if m, err := unix.Mmap(fd, 0, ringBytes, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED); err == nil {
			t.Ring = m
		}
	}
	if fd, err := unix.Open("/dev/dma_msi", unix.O_RDWR, 0); err == nil {
		t.msiFd = fd
		if m, err := unix.Mmap(fd, 0, ringBytes, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED); err == nil {
			t.MSI = m
		}
	}
	return t, nil
}

// Close unmaps regions and closes all device fds (releasing TCM).
func (t *TCM) Close() error {
	if t.Mem != nil {
		unix.Munmap(t.Mem)
		t.Mem = nil
	}
	t.Cores = nil
	if t.Ring != nil {
		unix.Munmap(t.Ring)
		t.Ring = nil
	}
	if t.MSI != nil {
		unix.Munmap(t.MSI)
		t.MSI = nil
	}
	for _, fd := range []int{t.tcmFd, t.aiFd, t.listFd, t.msiFd} {
		if fd >= 0 {
			unix.Close(fd)
		}
	}
	t.tcmFd, t.aiFd, t.listFd, t.msiFd = -1, -1, -1, -1
	return nil
}
