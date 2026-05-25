package main

import (
	"fmt"
	"runtime"
	"golang.org/x/sys/unix"
)

func main() {
	runtime.LockOSThread()
	for _, core := range []int{0, 4, 8, 12} {
		var set unix.CPUSet
		set.Zero()
		set.Set(core)
		err := unix.SchedSetaffinity(0, &set)
		fmt.Printf("Core %d: %v\n", core, err)
	}
}
