package nvidia

import (
	"math"
	"os"
	"testing"
)

// Run under Compute Sanitizer after explicit GPU recovery. Pure synthetic
// tensors exercise tile edges and buffer guards without reading held-out inputs.
func TestCompensatedProjectionCanariesGPU(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_BF16_PROJECTION") != "1" || os.Getenv("GO_PHERENCE_DISABLE_NVIDIA") != "" {
		t.Skip("opt-in GPU memory-safety probe")
	}
	if !Available() {
		t.Fatal("GPU explicitly requested but unavailable")
	}
	const guard = 64
	const sentinel = float32(-12345.25)
	guarded := func(n int) (*Buffer, *Buffer) {
		t.Helper()
		parent, e := Malloc(n + 2*guard)
		if e != nil {
			t.Fatal(e)
		}
		t.Cleanup(parent.Free)
		data := make([]float32, n+2*guard)
		for i := range data {
			data[i] = sentinel
		}
		if e = parent.Upload(data); e != nil {
			t.Fatal(e)
		}
		return parent, &Buffer{Ptr: parent.Ptr + guard*4, Size: n * 4}
	}
	check := func(parent *Buffer, n int) {
		t.Helper()
		data := make([]float32, n+2*guard)
		if e := parent.Download(data); e != nil {
			t.Fatal(e)
		}
		for i := 0; i < guard; i++ {
			if data[i] != sentinel || data[n+guard+i] != sentinel {
				t.Fatal("projection modified guard", i)
			}
		}
	}
	// Same aligned K/N dimensions as every Qwen projection, plus non-tile edges.
	for _, shape := range [][3]int{{1, 17, 33}, {15, 31, 65}, {16, 16, 2560}, {17, 1024, 2560}, {31, 4096, 2560}, {33, 9728, 2560}, {65, 2560, 4096}, {129, 2560, 9728}, {511, 17, 33}, {512, 19, 65}} {
		m, n, k := shape[0], shape[1], shape[2]
		ap, a := guarded(m * k)
		bp, b := guarded(k * n)
		cp, c := guarded(m * n)
		// Constant binary-exact operands allow a cheap oracle even at real width.
		av, bv := make([]float32, m*k), make([]float32, k*n)
		for i := range av {
			av[i] = .125
		}
		for i := range bv {
			bv[i] = .25
		}
		if e := a.Upload(av); e != nil {
			t.Fatal(e)
		}
		if e := b.Upload(bv); e != nil {
			t.Fatal(e)
		}
		if e := SgemmCompensated(m, n, k, 1, a, b, c); e != nil {
			t.Fatal(e)
		}
		if e := SyncErr(); e != nil {
			t.Fatal(e)
		}
		got := make([]float32, m*n)
		if e := c.Download(got); e != nil {
			t.Fatal(e)
		}
		want := float32(k) * .03125
		for i, v := range got {
			if math.Float32bits(v) != math.Float32bits(want) {
				t.Fatalf("shape=%v element=%d got=%g want=%g", shape, i, v, want)
			}
		}
		check(ap, m*k)
		check(bp, k*n)
		check(cp, m*n)
		// Release each shape promptly; test cleanup is idempotent.
		ap.Free()
		bp.Free()
		cp.Free()
	}
}
