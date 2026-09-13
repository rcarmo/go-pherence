package simd

import (
	"context"
	"errors"
	"fmt"
	"math"
	"runtime"
	"testing"
	"time"
)

type gemmPoolShape struct {
	name    string
	m, n, k int
	ldaPad  int
	ldbPad  int
	ldcPad  int
	alpha   float32
}

func makeGEMMPoolCase(shape gemmPoolShape) (a, w, c []float32, lda, ldb, ldc int) {
	lda = shape.k + shape.ldaPad
	ldb = shape.k + shape.ldbPad
	ldc = shape.n + shape.ldcPad
	a = make([]float32, (shape.m-1)*lda+shape.k)
	w = make([]float32, (shape.n-1)*ldb+shape.k)
	c = make([]float32, (shape.m-1)*ldc+shape.n)
	for i := range a {
		a[i] = float32(i%29-14) * 0.125
	}
	for i := range w {
		w[i] = float32(i%31-15) * 0.0625
	}
	for i := range c {
		c[i] = float32(i%17-8) * 0.25
	}
	return a, w, c, lda, ldb, ldc
}

func TestGEMMPoolExact(t *testing.T) {
	shapes := []gemmPoolShape{
		{name: "m126_n32_k63", m: 126, n: 32, k: 63, ldaPad: 3, ldbPad: 5, ldcPad: 7, alpha: 0.75},
		{name: "m126_n1024_k63", m: 126, n: 1024, k: 63, ldaPad: 1, ldbPad: 4, ldcPad: 9, alpha: -0.5},
		{name: "m128_n32_k127", m: 128, n: 32, k: 127, ldaPad: 2, ldbPad: 3, ldcPad: 5, alpha: 1.25},
		{name: "m128_n1024_k127", m: 128, n: 1024, k: 127, ldaPad: 4, ldbPad: 6, ldcPad: 11, alpha: -1},
	}
	for _, workers := range []int{1, 2, 4} {
		for _, mode := range []string{"streamed", "prepacked"} {
			for _, shape := range shapes {
				t.Run(fmt.Sprintf("%s/w%d/%s", shape.name, workers, mode), func(t *testing.T) {
					pool, err := NewGEMMPool(workers, shape.k)
					if err != nil {
						t.Fatalf("NewGEMMPool: %v", err)
					}
					defer func() {
						if err := pool.Close(); err != nil {
							t.Fatalf("Close: %v", err)
						}
					}()
					a, w, baseC, lda, ldb, ldc := makeGEMMPoolCase(shape)
					want := append([]float32(nil), baseC...)
					got := append([]float32(nil), baseC...)
					var packed []float32
					if mode == "prepacked" {
						packed, err = PackSgemmNTWeights(w, shape.n, shape.k, ldb)
						if err != nil {
							t.Fatalf("PackSgemmNTWeights: %v", err)
						}
						if !SgemmNTPrepackedTo(want, a, w, packed, shape.m, shape.n, shape.k, shape.alpha, lda, ldb, ldc) {
							t.Fatal("SgemmNTPrepackedTo rejected valid inputs")
						}
					} else {
						if !SgemmNTPackedTo(want, a, w, make([]float32, shape.k*gebpNR), shape.m, shape.n, shape.k, shape.alpha, lda, ldb, ldc) {
							t.Fatal("SgemmNTPackedTo rejected valid inputs")
						}
					}
					if err := pool.Run(context.Background(), got, a, w, packed, shape.m, shape.n, shape.k, shape.alpha, lda, ldb, ldc); err != nil {
						t.Fatalf("Run: %v", err)
					}
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("c[%d]=%08x want %08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
						}
					}
				})
			}
		}
	}
}

func TestGEMMPoolZeroAllocsNormal(t *testing.T) {
	shape := gemmPoolShape{name: "alloc", m: 128, n: 1024, k: 127, ldaPad: 3, ldbPad: 5, ldcPad: 7, alpha: 1}
	pool, err := NewGEMMPool(4, shape.k)
	if err != nil {
		t.Fatalf("NewGEMMPool: %v", err)
	}
	defer pool.Close()
	a, w, c, lda, ldb, ldc := makeGEMMPoolCase(shape)
	packed, err := PackSgemmNTWeights(w, shape.n, shape.k, ldb)
	if err != nil {
		t.Fatalf("PackSgemmNTWeights: %v", err)
	}
	if allocs := testing.AllocsPerRun(20, func() {
		clear(c)
		if err := pool.Run(context.Background(), c, a, w, nil, shape.m, shape.n, shape.k, shape.alpha, lda, ldb, ldc); err != nil {
			panic(err)
		}
	}); allocs != 0 {
		t.Fatalf("streamed allocs %g", allocs)
	}
	if allocs := testing.AllocsPerRun(20, func() {
		clear(c)
		if err := pool.Run(context.Background(), c, a, w, packed, shape.m, shape.n, shape.k, shape.alpha, lda, ldb, ldc); err != nil {
			panic(err)
		}
	}); allocs != 0 {
		t.Fatalf("prepacked allocs %g", allocs)
	}
}

func TestGEMMPoolPreCancelRetry(t *testing.T) {
	shape := gemmPoolShape{name: "cancel", m: 126, n: 1024, k: 63, ldaPad: 2, ldbPad: 4, ldcPad: 9, alpha: 0.5}
	pool, err := NewGEMMPool(2, shape.k)
	if err != nil {
		t.Fatalf("NewGEMMPool: %v", err)
	}
	defer pool.Close()
	a, w, baseC, lda, ldb, ldc := makeGEMMPoolCase(shape)
	got := append([]float32(nil), baseC...)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := pool.Run(ctx, got, a, w, nil, shape.m, shape.n, shape.k, shape.alpha, lda, ldb, ldc); !errors.Is(err, context.Canceled) {
		t.Fatalf("Run error=%v want context.Canceled", err)
	}
	for i := range got {
		if got[i] != baseC[i] {
			t.Fatalf("precancel mutated c[%d]", i)
		}
	}
	want := append([]float32(nil), baseC...)
	if !SgemmNTPackedTo(want, a, w, make([]float32, shape.k*gebpNR), shape.m, shape.n, shape.k, shape.alpha, lda, ldb, ldc) {
		t.Fatal("SgemmNTPackedTo rejected valid inputs")
	}
	if err := pool.Run(context.Background(), got, a, w, nil, shape.m, shape.n, shape.k, shape.alpha, lda, ldb, ldc); err != nil {
		t.Fatalf("retry Run: %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("retry c[%d]=%08x want %08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

func TestGEMMPoolRejectsBadInputsWithoutMutation(t *testing.T) {
	shape := gemmPoolShape{name: "bad", m: 128, n: 32, k: 63, ldaPad: 1, ldbPad: 2, ldcPad: 3, alpha: 1}
	a, w, baseC, lda, ldb, ldc := makeGEMMPoolCase(shape)
	t.Run("oversized-k", func(t *testing.T) {
		pool, err := NewGEMMPool(2, shape.k-1)
		if err != nil {
			t.Fatalf("NewGEMMPool: %v", err)
		}
		defer pool.Close()
		got := append([]float32(nil), baseC...)
		if err := pool.Run(context.Background(), got, a, w, nil, shape.m, shape.n, shape.k, shape.alpha, lda, ldb, ldc); !errors.Is(err, ErrGEMMPoolOversizedK) {
			t.Fatalf("Run error=%v want ErrGEMMPoolOversizedK", err)
		}
		for i := range got {
			if got[i] != baseC[i] {
				t.Fatalf("oversized-k mutated c[%d]", i)
			}
		}
	})
	t.Run("packed-short", func(t *testing.T) {
		pool, err := NewGEMMPool(2, shape.k)
		if err != nil {
			t.Fatal(err)
		}
		defer pool.Close()
		packed, err := PackSgemmNTWeights(w, shape.n, shape.k, ldb)
		if err != nil {
			t.Fatal(err)
		}
		got := append([]float32(nil), baseC...)
		if err := pool.Run(context.Background(), got, a, w, packed[:len(packed)-1], shape.m, shape.n, shape.k, shape.alpha, lda, ldb, ldc); !errors.Is(err, ErrGEMMPoolPackedSize) {
			t.Fatalf("short packed buffer: %v", err)
		}
		for i := range got {
			if got[i] != baseC[i] {
				t.Fatalf("mutated c[%d]", i)
			}
		}
	})
	t.Run("packed-alias", func(t *testing.T) {
		pool, err := NewGEMMPool(4, shape.k)
		if err != nil {
			t.Fatalf("NewGEMMPool: %v", err)
		}
		defer pool.Close()
		cLen := (shape.m-1)*ldc + shape.n
		_, packedLen, ok := checkedSgemmNTFullPanelLayout(shape.n, shape.k)
		if !ok {
			t.Fatal("unexpected packed length overflow")
		}
		storage := make([]float32, cLen+packedLen+64)
		for i := range storage {
			storage[i] = float32(i + 1)
		}
		c := storage[8 : 8+cLen]
		packed := storage[16 : 16+packedLen]
		before := append([]float32(nil), storage...)
		if err := pool.Run(context.Background(), c, a, w, packed, shape.m, shape.n, shape.k, shape.alpha, lda, ldb, ldc); !errors.Is(err, ErrGEMMPoolPackedAlias) {
			t.Fatalf("Run error=%v want ErrGEMMPoolPackedAlias", err)
		}
		for i := range storage {
			if storage[i] != before[i] {
				t.Fatalf("packed-alias mutated storage[%d]", i)
			}
		}
	})
}

func TestGEMMPoolCloseIdempotentAndConcurrent(t *testing.T) {
	shape := gemmPoolShape{name: "close", m: 128, n: 1024, k: 127, ldaPad: 0, ldbPad: 0, ldcPad: 0, alpha: 1}
	pool, err := NewGEMMPool(4, shape.k)
	if err != nil {
		t.Fatalf("NewGEMMPool: %v", err)
	}
	a, w, c, lda, ldb, ldc := makeGEMMPoolCase(shape)
	runDone := make(chan error, 1)
	closeDone := make(chan error, 1)
	ctx := &gatedGEMMContext{Context: context.Background(), gateCall: 3, entered: make(chan struct{}), release: make(chan struct{})}
	go func() {
		runDone <- pool.Run(ctx, c, a, w, nil, shape.m, shape.n, shape.k, shape.alpha, lda, ldb, ldc)
	}()
	<-ctx.entered // Run owns runMu and has submitted a worker.
	go func() {
		closeDone <- pool.Close()
	}()
	deadline := time.Now().Add(5 * time.Second)
	for {
		pool.stateMu.Lock()
		closing := pool.closing
		pool.stateMu.Unlock()
		if closing {
			break
		}
		if time.Now().After(deadline) {
			close(ctx.release)
			t.Fatal("Close did not start")
		}
		runtime.Gosched()
	}
	select {
	case <-closeDone:
		close(ctx.release)
		t.Fatal("Close returned during active Run")
	default:
	}
	close(ctx.release)
	select {
	case err := <-runDone:
		if err != nil && !errors.Is(err, ErrGEMMPoolClosed) {
			t.Fatalf("Run error: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Run deadlocked")
	}
	select {
	case err := <-closeDone:
		if err != nil {
			t.Fatalf("Close error: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Close deadlocked")
	}
	if err := pool.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if err := pool.Run(context.Background(), c, a, w, nil, shape.m, shape.n, shape.k, shape.alpha, lda, ldb, ldc); !errors.Is(err, ErrGEMMPoolClosed) {
		t.Fatalf("Run after Close error=%v want ErrGEMMPoolClosed", err)
	}
}

func TestGEMMPoolNoGoroutineLeak(t *testing.T) {
	before := runtime.NumGoroutine()
	pool, err := NewGEMMPool(4, 63)
	if err != nil {
		t.Fatalf("NewGEMMPool: %v", err)
	}
	if err := pool.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for runtime.NumGoroutine() > before+1 && time.Now().Before(deadline) {
		runtime.Gosched()
		time.Sleep(time.Millisecond)
	}
	if got := runtime.NumGoroutine(); got > before+1 {
		t.Fatalf("goroutines after Close=%d before=%d", got, before)
	}
}

// Err is called by the Run coordinator only. Gate after the first submitted
// job to pin cancellation/Close overlap without scheduler-dependent sleeps.
type gatedGEMMContext struct {
	context.Context
	calls, gateCall  int
	entered, release chan struct{}
}

func (c *gatedGEMMContext) Err() error {
	c.calls++
	if c.calls == c.gateCall {
		close(c.entered)
		<-c.release
	}
	return c.Context.Err()
}

func TestGEMMPoolMidRunCancelDrainsAndRetries(t *testing.T) {
	shape := gemmPoolShape{name: "cancel", m: 128, n: 32, k: 63, alpha: 1}
	a, w, base, lda, ldb, ldc := makeGEMMPoolCase(shape)
	for _, prepacked := range []bool{false, true} {
		pool, err := NewGEMMPool(4, shape.k)
		if err != nil {
			t.Fatal(err)
		}
		defer pool.Close()
		var packed []float32
		if prepacked {
			packed, err = PackSgemmNTWeights(w, shape.n, shape.k, ldb)
			if err != nil {
				t.Fatal(err)
			}
		}
		underlying, cancel := context.WithCancel(context.Background())
		ctx := &gatedGEMMContext{Context: underlying, gateCall: 3, entered: make(chan struct{}), release: make(chan struct{})}
		got := append([]float32(nil), base...)
		done := make(chan error, 1)
		go func() { done <- pool.Run(ctx, got, a, w, packed, shape.m, shape.n, shape.k, 1, lda, ldb, ldc) }()
		<-ctx.entered
		cancel()
		close(ctx.release)
		if err := <-done; !errors.Is(err, context.Canceled) {
			t.Fatalf("cancel: %v", err)
		}
		// Run must drain the already-submitted worker before returning. Copy/reuse
		// immediately; the race detector also checks no late writes survive.
		want := append([]float32(nil), base...)
		scratch := make([]float32, shape.k*gebpNR)
		if !SgemmNTPackedTo(want, a, w, scratch, shape.m, shape.n, shape.k, 1, lda, ldb, ldc) {
			t.Fatal("serial")
		}
		copy(got, base)
		if err := pool.Run(context.Background(), got, a, w, packed, shape.m, shape.n, shape.k, 1, lda, ldb, ldc); err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("retry c[%d]", i)
			}
		}
	}
}
