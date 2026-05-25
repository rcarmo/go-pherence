package main
import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"syscall"
	"golang.org/x/sys/unix"
)
func main() {
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			runtime.LockOSThread()
			tid := syscall.Gettid()
			// Write TID to /proc/set_ai_thread
			f, err := os.OpenFile("/proc/set_ai_thread", os.O_WRONLY, 0)
			if err != nil { fmt.Printf("Worker %d: open failed: %v\n", id, err); return }
			_, err = fmt.Fprintf(f, "%d", tid)
			f.Close()
			if err != nil { fmt.Printf("Worker %d: write failed: %v\n", id, err); return }
			// Now try pinning to core 8+id
			var set unix.CPUSet
			set.Zero()
			set.Set(8 + id)
			err = unix.SchedSetaffinity(0, &set)
			if err == nil {
				fmt.Printf("Worker %d: pinned to core %d!\n", id, 8+id)
			} else {
				fmt.Printf("Worker %d: pin failed: %v\n", id, err)
			}
		}(i)
	}
	wg.Wait()
}
