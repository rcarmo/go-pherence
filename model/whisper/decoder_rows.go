package whisper

import (
	simdrt "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"os"
	"runtime"
	"sync"
)

// Explicit scheduling-only experiment. Default decoder projections stay serial;
// int8 dispatch in linearInto remains separate and takes precedence.
const envDecoderParallelRows = "GO_PHERENCE_WHISPER_DECODER_PARALLEL_ROWS"

func decoderRowWorkers(outDim int) int {
	if os.Getenv(envDecoderParallelRows) != "1" || outDim < 512 {
		return 1
	}
	return max(1, min(min(linearWorkers, runtime.GOMAXPROCS(0)), outDim))
}

// decoderLinearRowsInto partitions independent output cells, never a dot's
// reduction. Inputs/weights/bias must be immutable and not overlap out, as in
// decoder scratch ownership. Each worker uses the exact existing Sdot+bias
// expression; all workers join before the caller continues or observes cancel.
func decoderLinearRowsInto(out, x, weight, bias []float32, inDim, outDim, workers int) {
	// Preflight extents synchronously: malformed input must not panic in a worker.
	_ = out[:outDim]
	_ = x[:inDim]
	_ = weight[:inDim*outDim]
	compute := func(start, end int) {
		for o := start; o < end; o++ {
			wOff := o * inDim
			sum := simdrt.Sdot(x[:inDim], weight[wOff:wOff+inDim])
			if bias != nil && o < len(bias) {
				sum += bias[o]
			}
			out[o] = sum
		}
	}
	workers = max(1, min(workers, outDim))
	if workers <= 1 {
		compute(0, outDim)
		return
	}
	chunk := (outDim + workers - 1) / workers
	var wg sync.WaitGroup
	for start := 0; start < outDim; start += chunk {
		end := min(start+chunk, outDim)
		wg.Add(1)
		go func(a, b int) { defer wg.Done(); compute(a, b) }(start, end)
	}
	wg.Wait()
}
