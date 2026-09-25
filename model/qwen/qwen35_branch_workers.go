package qwen

import (
	"fmt"
	"runtime"
	"sync"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// Workers live for one locked Forward, not for every projection or the model
// lifetime. No goroutine remains parked when an idle scorer becomes unreachable.
type branchProjectionJob struct {
	index                  int
	dst, x, w, packed      []float32
	rows, cols, in, stride int
}

func (s *Qwen35SIMDBranch) startProjectionWorkers() func() {
	workers := min(runtime.GOMAXPROCS(0), 6)
	if workers < 2 {
		return func() {}
	}
	s.jobs = make(chan branchProjectionJob, workers)
	var done sync.WaitGroup
	done.Add(workers)
	for range workers {
		go func() {
			defer done.Done()
			for j := range s.jobs {
				s.jobFailed[j.index] = !simd.SgemmNTPrepackedTo(j.dst, j.x, j.w, j.packed, j.rows, j.cols, j.in, 1, j.in, j.in, j.stride)
				s.jobWait.Done()
			}
		}()
	}
	return func() { close(s.jobs); done.Wait(); s.jobs = nil }
}

// Fixed-shape branch GQA with caller-owned scores/output. No padding token is
// visible: the caller slices K/V and scores to this node's ancestor boundary.
func qwen35BranchAttentionInto(out, scores, q, k, v []float32) error {
	n := len(scores)
	if n < 1 || len(out) != 2048 || len(q) != 2048 || len(k) != n*512 || len(v) != n*512 {
		return fmt.Errorf("qwen: invalid branch attention buffers")
	}
	clear(out)
	for h := 0; h < 8; h++ {
		qh, oh := q[h*256:(h+1)*256], out[h*256:(h+1)*256]
		kvh := h / 4 * 256
		for t := 0; t < n; t++ {
			scores[t] = simd.Sdot(qh, k[t*512+kvh:t*512+kvh+256]) * 0.0625
		}
		if !simd.SoftmaxInPlace(scores) {
			return fmt.Errorf("qwen: nonfinite branch attention")
		}
		for t, w := range scores {
			simd.Saxpy(w, v[t*512+kvh:t*512+kvh+256], oh)
		}
	}
	return nil
}
