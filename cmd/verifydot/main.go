package main

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"syscall"
	"time"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"github.com/rcarmo/go-pherence/backends/spacemit/tcm"
)

func main() {
	runtime.LockOSThread()
	tid := syscall.Gettid()
	f, _ := os.OpenFile("/proc/set_ai_thread", os.O_WRONLY, 0)
	f.Write([]byte(strconv.Itoa(tid)))
	f.Close()

	K := 2048
	
	// Small test: just 1 group
	wI8 := make([]int8, 4*K)
	for i := range wI8 { wI8[i] = int8(i%15 + 1) }
	wPacked := ime2.PackTiles(wI8, 4, K)
	
	actI8 := make([]int8, 4*K)
	for i := range actI8 { actI8[i] = 10 }
	actPacked := ime2.PackTiles(actI8, 4, K)

	// Stage to TCM
	tcmDev, _ := tcm.Open()
	tcmSlice := tcmDev.Slice(0)
	tileBytes := (K / 8) * 32 // 8192
	fmt.Printf("Copying %d bytes to TCM\n", tileBytes)
	copy(tcmSlice[:tileBytes], unsafe.Slice((*byte)(unsafe.Pointer(&wPacked[0])), tileBytes))
	
	// Verify TCM has data
	fmt.Printf("TCM[0:8]: %v\n", tcmSlice[:8])
	fmt.Printf("wPacked[0:8]: %v\n", unsafe.Slice((*byte)(unsafe.Pointer(&wPacked[0])), 8))
	
	// vmadot from DRAM
	var accDRAM [16]int32
	ime2.VmadotKLoop((*byte)(unsafe.Pointer(&wPacked[0])), (*byte)(unsafe.Pointer(&actPacked[0])), &accDRAM[0], K)
	fmt.Printf("DRAM result: %d\n", accDRAM[0])
	
	// vmadot from TCM (using slice element pointer)
	var accTCM [16]int32
	tcmPtr := unsafe.Pointer(&tcmSlice[0])
	fmt.Printf("TCM ptr: %p\n", tcmPtr)
	ime2.VmadotKLoop((*byte)(tcmPtr), (*byte)(unsafe.Pointer(&actPacked[0])), &accTCM[0], K)
	fmt.Printf("TCM result: %d\n", accTCM[0])
	
	if accDRAM[0] == accTCM[0] {
		fmt.Println("MATCH! vmadot from TCM works on AI cores!")
	}
	
	// Benchmark
	const iters = 5000
	t0 := time.Now()
	for i := 0; i < iters; i++ {
		var acc [16]int32
		ime2.VmadotKLoop((*byte)(unsafe.Pointer(&wPacked[0])), (*byte)(unsafe.Pointer(&actPacked[0])), &acc[0], K)
	}
	dramTime := time.Since(t0)
	
	t1 := time.Now()
	for i := 0; i < iters; i++ {
		var acc [16]int32
		ime2.VmadotKLoop((*byte)(tcmPtr), (*byte)(unsafe.Pointer(&actPacked[0])), &acc[0], K)
	}
	tcmTime := time.Since(t1)
	
	fmt.Printf("DRAM: %v/call\n", dramTime/iters)
	fmt.Printf("TCM:  %v/call\n", tcmTime/iters)
	fmt.Printf("Speedup: %.2fx\n", float64(dramTime)/float64(tcmTime))
}
