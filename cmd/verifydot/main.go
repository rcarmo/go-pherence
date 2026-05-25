package main

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"syscall"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
)

func main() {
	runtime.LockOSThread()
	f, _ := os.OpenFile("/proc/set_ai_thread", os.O_WRONLY, 0)
	f.Write([]byte(strconv.Itoa(syscall.Gettid())))
	f.Close()

	K := 32
	var acc [64]int32

	// Test: all weights=1, all act=1, K=32
	wI8 := make([]int8, 4*K)
	for i := range wI8 { wI8[i] = 1 }
	wp := ime2.PackTiles1024(wI8, 4, K)
	aI8 := make([]int8, K)
	for i := range aI8 { aI8[i] = 1 }
	ap := ime2.BroadcastPack1024(aI8, K)

	ime2.VmadotKLoop1024((*byte)(unsafe.Pointer(&wp[0])), (*byte)(unsafe.Pointer(&ap[0])), &acc[0], K)

	// Expected full dot product = 1*32 = 32 for C[0][0]
	fmt.Println("K=32, wt=1, act=1. Expected C[0][0]=32. Testing copy hypothesis:")
	fmt.Printf("  acc[0]+acc[16]+acc[32]+acc[48] = %d\n", acc[0]+acc[16]+acc[32]+acc[48])
	fmt.Printf("  acc[0]+acc[8] = %d\n", acc[0]+acc[8])
	fmt.Printf("  acc[0]+acc[1] = %d\n", acc[0]+acc[1])
	fmt.Println()
	fmt.Println("All 64 values:")
	for i := 0; i < 64; i++ {
		if acc[i] != 0 {
			fmt.Printf("  [%d]=%d", i, acc[i])
		}
	}
	fmt.Println()

	// Test 2: different row values to identify row mapping
	for i := range acc { acc[i] = 0 }
	for i := range wI8 { wI8[i] = 0 }
	// Only row 0 has value 1
	for c := 0; c < K; c++ { wI8[0*K+c] = 1 }
	wp2 := ime2.PackTiles1024(wI8, 4, K)
	ime2.VmadotKLoop1024((*byte)(unsafe.Pointer(&wp2[0])), (*byte)(unsafe.Pointer(&ap[0])), &acc[0], K)
	fmt.Println("\nOnly wt_row0=1, rest=0. Expected: only C[0][x] nonzero")
	for i := 0; i < 64; i++ {
		if acc[i] != 0 { fmt.Printf("  [%d]=%d", i, acc[i]) }
	}
	fmt.Println()

	// Test 3: only act_row0 has value
	for i := range acc { acc[i] = 0 }
	for i := range wI8 { wI8[i] = 1 } // all weights = 1
	wp3 := ime2.PackTiles1024(wI8, 4, K)
	aI8_2 := make([]int8, K)
	// only first 8 elements of act are 1, rest are 0
	for i := 0; i < 8; i++ { aI8_2[i] = 1 }
	ap2 := ime2.BroadcastPack1024(aI8_2, K)
	ime2.VmadotKLoop1024((*byte)(unsafe.Pointer(&wp3[0])), (*byte)(unsafe.Pointer(&ap2[0])), &acc[0], K)
	fmt.Println("\nact=[1]*8+[0]*24, wt=1. Expected partial dot = 8 per row")
	for i := 0; i < 64; i++ {
		if acc[i] != 0 { fmt.Printf("  [%d]=%d", i, acc[i]) }
	}
	fmt.Println()
}
