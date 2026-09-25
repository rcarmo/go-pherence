//go:build linux && (amd64 || arm64)

package nvidia

import (
	"runtime"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
)

type launchBindingObservation struct {
	fn           CUfunction
	gridX        uint32
	gridY        uint32
	gridZ        uint32
	blockX       uint32
	blockY       uint32
	blockZ       uint32
	sharedMem    uint32
	stream       uintptr
	kernelParams unsafe.Pointer
	extra        unsafe.Pointer
	paramTable   [3]unsafe.Pointer
	devicePtr    CUdeviceptr
	width        uint32
	word         uintptr
	extraWords   [2]uintptr
}

func TestFixedCUDAKernelLauncherABI(t *testing.T) {
	const (
		wantResult CUresult = 0x7f123456
		paramCount          = 3
		extraCount          = 2
	)

	deviceArg := new(CUdeviceptr)
	*deviceArg = CUdeviceptr(0xF0E1D2C3B4A59687)
	widthArg := new(uint32)
	*widthArg = uint32(0x89ABCDEF)
	wordArg := new(uintptr)
	*wordArg = uintptr(0x0123456789ABCDEF)

	kernelTable := &[paramCount]unsafe.Pointer{
		unsafe.Pointer(deviceArg),
		unsafe.Pointer(widthArg),
		unsafe.Pointer(wordArg),
	}
	kernelParams := unsafe.Pointer(&kernelTable[0])

	extraWords := &[extraCount]uintptr{
		uintptr(0xA1B2C3D4E5F60718),
		uintptr(0x192A3B4C5D6E7F80),
	}
	extra := unsafe.Pointer(&extraWords[0])

	fn := CUfunction(0xFEDCBA9876543210)
	gridX := uint32(0x89ABCDEF)
	gridY := uint32(0x76543210)
	gridZ := uint32(0xFFFFFFFF)
	blockX := uint32(0x80000001)
	blockY := uint32(0xF0E1D2C3)
	blockZ := uint32(0x7FFFFFFF)
	sharedMem := uint32(0xA5A5A5A5)
	stream := uintptr(0xDEADBEEFCAFEBABE)

	want := launchBindingObservation{
		fn:           fn,
		gridX:        gridX,
		gridY:        gridY,
		gridZ:        gridZ,
		blockX:       blockX,
		blockY:       blockY,
		blockZ:       blockZ,
		sharedMem:    sharedMem,
		stream:       stream,
		kernelParams: kernelParams,
		extra:        extra,
		paramTable: [paramCount]unsafe.Pointer{
			kernelTable[0],
			kernelTable[1],
			kernelTable[2],
		},
		devicePtr:  *deviceArg,
		width:      *widthArg,
		word:       *wordArg,
		extraWords: [extraCount]uintptr{extraWords[0], extraWords[1]},
	}

	var calls int
	var got [2]launchBindingObservation
	callback := purego.NewCallback(func(fn CUfunction, gx, gy, gz, bx, by, bz, sm uint32, stream uintptr, kernelParams, extra unsafe.Pointer) CUresult {
		if calls >= len(got) {
			panic("unexpected callback count")
		}
		runtime.GC()
		params := unsafe.Slice((*unsafe.Pointer)(kernelParams), paramCount)
		extraWords := unsafe.Slice((*uintptr)(extra), extraCount)
		got[calls] = launchBindingObservation{
			fn:           fn,
			gridX:        gx,
			gridY:        gy,
			gridZ:        gz,
			blockX:       bx,
			blockY:       by,
			blockZ:       bz,
			sharedMem:    sm,
			stream:       stream,
			kernelParams: kernelParams,
			extra:        extra,
			paramTable: [paramCount]unsafe.Pointer{
				params[0],
				params[1],
				params[2],
			},
			devicePtr: *(*CUdeviceptr)(params[0]),
			width:     *(*uint32)(params[1]),
			word:      *(*uintptr)(params[2]),
			extraWords: [extraCount]uintptr{
				extraWords[0],
				extraWords[1],
			},
		}
		calls++
		return wantResult
	})

	var baseline func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult
	purego.RegisterFunc(&baseline, callback)
	launch := fixedCUDAKernelLauncher(callback)

	if gotResult := baseline(fn, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMem, stream, kernelParams, extra); gotResult != wantResult {
		t.Fatalf("baseline return = %#x, want %#x", gotResult, wantResult)
	}
	if gotResult := launch(fn, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMem, stream, kernelParams, extra); gotResult != wantResult {
		t.Fatalf("fixed launcher return = %#x, want %#x", gotResult, wantResult)
	}

	runtime.KeepAlive(deviceArg)
	runtime.KeepAlive(widthArg)
	runtime.KeepAlive(wordArg)
	runtime.KeepAlive(kernelTable)
	runtime.KeepAlive(extraWords)

	if calls != 2 {
		t.Fatalf("callback count = %d, want 2", calls)
	}
	if got[0] != want {
		t.Fatalf("baseline observation mismatch\n got: %#v\nwant: %#v", got[0], want)
	}
	if got[1] != got[0] {
		t.Fatalf("fixed launcher ABI mismatch with RegisterFunc baseline\nbaseline: %#v\n   fixed: %#v", got[0], got[1])
	}
}
