// npu-tcm validates the pure-Go TCM substrate against the SpaceMIT NPU:
// opens the device, prints geometry, acquires cores, and round-trips TCM data.
package main

import (
	"fmt"
	"os"

	"github.com/rcarmo/go-pherence/npu"
)

func main() {
	if !npu.Available() {
		fmt.Fprintln(os.Stderr, "NPU /dev/tcm not available")
		os.Exit(1)
	}
	t, err := npu.Open()
	if err != nil {
		fmt.Fprintln(os.Stderr, "open:", err)
		os.Exit(1)
	}
	defer t.Close()
	fmt.Printf("TCM: %d cores x %d bytes (%.0f KiB/core, %.1f MiB), acquired %d %v\n",
		t.NumCores, t.BlockSize, float64(t.BlockSize)/1024,
		float64(t.BlockSize*t.NumCores)/1048576, len(t.Acquired), t.Acquired)
	fmt.Printf("rings: ai_dma list=%v msi=%v\n", t.Ring != nil, t.MSI != nil)

	// round-trip each core's TCM via the mapped region
	for c := 0; c < t.NumCores; c++ {
		t.Cores[c][0] = byte(0xA0 + c)
		t.Cores[c][1] = byte(c)
	}
	ok := true
	fmt.Printf("TCM round-trip core[0]: ")
	for c := 0; c < t.NumCores; c++ {
		fmt.Printf("%02x ", t.Cores[c][0])
		if t.Cores[c][0] != byte(0xA0+c) {
			ok = false
		}
	}
	fmt.Printf("\nread-back %v\n", map[bool]string{true: "OK", false: "MISMATCH"}[ok])
}
