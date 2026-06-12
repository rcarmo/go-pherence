// Command testi8i4 exercises the canonical IME2 i8×i4 M1/N32 kernel
// (ime2.K3I8I4M1) in isolation: it registers an AI worker thread, pins to
// core 8, and prints which output lanes light up for each weight offset.
// Useful for kernel bring-up/validation. The kernel itself lives in
// backends/spacemit/ime2 — this command keeps no private copy.
package main

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"syscall"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"golang.org/x/sys/unix"
)

func main() {
	runtime.LockOSThread()
	if f, e := os.OpenFile("/proc/set_ai_thread", os.O_WRONLY, 0); e == nil {
		_, _ = f.Write([]byte(strconv.Itoa(syscall.Gettid())))
		f.Close()
	}
	var set unix.CPUSet
	set.Zero()
	set.Set(8)
	unix.SchedSetaffinity(0, &set)
	for off := 64; off < 128; off++ {
		a := make([]byte, 38)
		b := make([]byte, 640)
		c := make([]float32, 32)
		*(*float32)(unsafe.Pointer(&a[0])) = 1
		*(*int16)(unsafe.Pointer(&a[4])) = -1
		for i := 0; i < 32; i++ {
			b[i*2] = 0
			b[i*2+1] = 0x3c
		}
		b[off] = 1
		ime2.K3I8I4M1((*byte)(unsafe.Pointer(&a[0])), (*byte)(unsafe.Pointer(&b[0])), &c[0], 1, 32)
		printed := false
		for i, v := range c {
			if v != 0 {
				if !printed {
					fmt.Printf("off%d:", off)
					printed = true
				}
				fmt.Printf(" %d=%.0f", i, v)
			}
		}
		if printed {
			fmt.Println()
		}
	}
}
