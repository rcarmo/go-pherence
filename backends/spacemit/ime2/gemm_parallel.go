package ime2

import (
	"runtime"
	"sync"
	"unsafe"

	"golang.org/x/sys/unix"
)

// GemmINT8PackedParallel performs C[M×N] = A_packed * B_packed^T using vmadot
// with multiple goroutines pinned to X100 cores.
// Each goroutine handles a slice of M rows.
func GemmINT8PackedParallel(M, N, K int, Apacked, Bpacked []int8, C []int32, nThreads int) {
	if M%4 != 0 || N%4 != 0 || K%8 != 0 {
		panic("ime2: dimensions must be multiples of 4/4/8")
	}
	if nThreads <= 1 {
		GemmINT8Packed(M, N, K, Apacked, Bpacked, C)
		return
	}

	tilesPerRow := K / 8

	var wg sync.WaitGroup
	rowsPerThread := ((M / 4) / nThreads) * 4 // round down to multiple of 4
	if rowsPerThread < 4 {
		rowsPerThread = 4
	}

	for t := 0; t < nThreads; t++ {
		iStart := t * rowsPerThread
		iEnd := iStart + rowsPerThread
		if t == nThreads-1 {
			iEnd = M // last thread takes remainder
		}
		if iStart >= M {
			break
		}

		wg.Add(1)
		go func(iStart, iEnd, coreID int) {
			defer wg.Done()

			// Pin goroutine to a specific OS thread
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()

			// Pin thread to X100 core (cores 0-7)
			var cpuSet unix.CPUSet
			cpuSet.Zero()
			cpuSet.Set(coreID)
			unix.SchedSetaffinity(0, &cpuSet)

			// Process assigned rows
			for i := iStart; i < iEnd; i += 4 {
				aBase := (i / 4) * tilesPerRow * 32
				for j := 0; j < N; j += 4 {
					bBase := (j / 4) * tilesPerRow * 32

					var acc [16]int32
					vmadotKLoop(
						(*byte)(unsafe.Pointer(&Apacked[aBase])),
						(*byte)(unsafe.Pointer(&Bpacked[bBase])),
						&acc[0],
						K,
					)

					// Write output tile
					for r := 0; r < 4; r++ {
						for c := 0; c < 4; c++ {
							C[(i+r)*N+(j+c)] = acc[r*4+c]
						}
					}
				}
			}
		}(iStart, iEnd, t%8)
	}
	wg.Wait()
}
