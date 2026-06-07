package aipool

import (
	"os"
	"runtime"
	"strconv"
	"syscall"

	"golang.org/x/sys/unix"
)

// RegisterAIThread grants the current goroutine access to cores 8-15
// by writing its TID to /proc/set_ai_thread, then pins to coreID.
func RegisterAIThread(coreID int) {
	runtime.LockOSThread()
	tid := syscall.Gettid()
	if f, err := os.OpenFile("/proc/set_ai_thread", os.O_WRONLY, 0); err == nil {
		f.Write([]byte(strconv.Itoa(tid)))
		f.Close()
	}
	var set unix.CPUSet
	set.Zero()
	set.Set(coreID)
	unix.SchedSetaffinity(0, &set)
}
