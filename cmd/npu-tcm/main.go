// npu-tcm validates the pure-Go TCM substrate: opens /dev/tcm, prints geometry,
// acquires cores, and round-trips data through each core's SRAM block.
//
// NOTE: TCM is uncached device memory — usable for bulk DMA staging, not as a
// CPU/RVV compute scratchpad (see research/npu-whisper/TCM_DEAD_END.md).
package main

import (
	"fmt"
	"os"

	"github.com/rcarmo/go-pherence/backends/spacemit/tcm"
)

func main() {
	if !tcm.IsAvailable() {
		fmt.Fprintln(os.Stderr, "NPU /dev/tcm not available")
		os.Exit(1)
	}
	t, err := tcm.Open()
	if err != nil {
		fmt.Fprintln(os.Stderr, "open:", err)
		os.Exit(1)
	}
	defer t.Close()

	acquired := 0
	for c := 0; c < tcm.BlockCount; c++ {
		if err := t.Acquire(c); err == nil {
			acquired++
		}
	}
	fmt.Printf("TCM: %d blocks x %d bytes (%.1f MiB), acquired %d/%d\n",
		tcm.BlockCount, tcm.BlockSize, float64(tcm.TotalSize)/1048576, acquired, tcm.BlockCount)

	ok := true
	for c := 0; c < tcm.BlockCount; c++ {
		t.Slice(c)[0] = byte(0xA0 + c)
	}
	fmt.Printf("TCM round-trip block[0]: ")
	for c := 0; c < tcm.BlockCount; c++ {
		v := t.Slice(c)[0]
		fmt.Printf("%02x ", v)
		if v != byte(0xA0+c) {
			ok = false
		}
	}
	fmt.Printf("\nread-back %v\n", map[bool]string{true: "OK", false: "MISMATCH"}[ok])
}
