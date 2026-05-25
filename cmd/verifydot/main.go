package main

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"syscall"
	"unsafe"
)

func main() {
	runtime.LockOSThread()
	tid := syscall.Gettid()
	f, _ := os.OpenFile("/proc/set_ai_thread", os.O_WRONLY, 0)
	f.Write([]byte(strconv.Itoa(tid)))
	f.Close()
	
	// Read VLEN by executing vsetvli and checking the result
	// vsetvli with e8,m1 gives vl = VLEN/8
	// Inline asm to get vl... actually cant do inline asm in Go.
	// Alternative: just try a vle8 with large vl and see how many bytes are loaded
	buf := make([]byte, 128)
	for i := range buf { buf[i] = byte(i+1) }
	out := make([]byte, 128)
	
	// Use our existing vmadotSS4x8 which does vsetvli at the start
	// It sets vl=32 (for VLEN=256). If VLEN is different on core 8, it would set differently.
	// Actually vsetvli t0, zero, e8, m1 gives vl=VLEN/8.
	// For VLEN=256: vl=32. For VLEN=512: vl=64.
	// Our vmadotKLoop loads 32 bytes per tile. If VLEN=512, the load reads 64 bytes!
	// THATS THE BUG! A100 cores might have VLEN=512 not 256!
	
	fmt.Printf("Testing from AI core...\n")
	fmt.Printf("buf at %p, out at %p\n", unsafe.Pointer(&buf[0]), unsafe.Pointer(&out[0]))
	// Just print that we got here
	fmt.Println("Need to check VLEN on these cores!")
}
