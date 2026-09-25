package mojev

import (
	"slices"
	"testing"
)

func gpuTreeField(dst []uint32, field int) []uint32 {
	result := make([]uint32, len(dst)/4)
	for i := range result {
		result[i] = dst[i*4+field]
	}
	return result
}

func filledUint32(n int, value uint32) []uint32 {
	dst := make([]uint32, n)
	for i := range dst {
		dst[i] = value
	}
	return dst
}

func TestFillGPUTreeMetadataExact(t *testing.T) {
	const root = ^uint32(0)
	ends := [...]int{5, 8, 10}
	dst := make([]uint32, 10*4)
	if err := fillGPUTree(dst, 10, 2, 2, ends[:]); err != nil {
		t.Fatal(err)
	}
	if got := gpuTreeField(dst, 0); !slices.Equal(got, []uint32{root, 0, 1, 2, 3, 3, 5, 6, 3, 8}) {
		t.Fatalf("parent = %v", got)
	}
	if got := gpuTreeField(dst, 1); !slices.Equal(got, []uint32{0, 1, 2, 3, 4, 4, 5, 6, 4, 5}) {
		t.Fatalf("local position = %v", got)
	}
	if got := gpuTreeField(dst, 2); !slices.Equal(got, []uint32{0, 0, 0, 0, 4, 5, 5, 5, 8, 8}) {
		t.Fatalf("start = %v", got)
	}
	if got := gpuTreeField(dst, 3); !slices.Equal(got, []uint32{2, 2, 4, 4, 5, 8, 8, 8, 10, 10}) {
		t.Fatalf("end = %v", got)
	}
}

func TestFillGPUTreeRejectsInvalidGeometryWithoutMutation(t *testing.T) {
	tooManyEnds := make([]int, 65)
	for i := range tooManyEnds {
		tooManyEnds[i] = i + 1
	}
	const sentinel = uint32(0xdecafbad)
	tests := []struct {
		name   string
		dstLen int
		n      int
		ns     int
		nq     int
		ends   []int
	}{
		{name: "n_too_small", dstLen: 2 * 4, n: 2, ns: 1, nq: 1, ends: []int{2}},
		{name: "n_too_large", dstLen: 513 * 4, n: 513, ns: 1, nq: 1, ends: []int{513}},
		{name: "dst_short", dstLen: 10*4 - 1, n: 10, ns: 2, nq: 2, ends: []int{5, 8, 10}},
		{name: "dst_long", dstLen: 10*4 + 1, n: 10, ns: 2, nq: 2, ends: []int{5, 8, 10}},
		{name: "ns_zero", dstLen: 10 * 4, n: 10, ns: 0, nq: 2, ends: []int{5, 8, 10}},
		{name: "ns_too_large", dstLen: 10 * 4, n: 10, ns: 10, nq: 1, ends: []int{10}},
		{name: "nq_zero", dstLen: 10 * 4, n: 10, ns: 2, nq: 0, ends: []int{5, 8, 10}},
		{name: "nq_too_large", dstLen: 10 * 4, n: 10, ns: 2, nq: 8, ends: []int{10}},
		{name: "nil_ends", dstLen: 10 * 4, n: 10, ns: 2, nq: 2, ends: nil},
		{name: "too_many_ends", dstLen: 10 * 4, n: 10, ns: 2, nq: 2, ends: tooManyEnds},
		{name: "decreasing_ends", dstLen: 10 * 4, n: 10, ns: 2, nq: 2, ends: []int{6, 5, 10}},
		{name: "repeated_ends", dstLen: 10 * 4, n: 10, ns: 2, nq: 2, ends: []int{5, 5, 10}},
		{name: "missing_tail", dstLen: 10 * 4, n: 10, ns: 2, nq: 2, ends: []int{5, 8, 9}},
		{name: "end_over_n", dstLen: 10 * 4, n: 10, ns: 2, nq: 2, ends: []int{5, 8, 11}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dst := filledUint32(tc.dstLen, sentinel)
			want := slices.Clone(dst)
			if err := fillGPUTree(dst, tc.n, tc.ns, tc.nq, tc.ends); err == nil {
				t.Fatal("accepted invalid geometry")
			}
			if !slices.Equal(dst, want) {
				t.Fatalf("mutated dst = %v want %v", dst, want)
			}
		})
	}
}

func TestFillGPUTreeWarmAllocations(t *testing.T) {
	var dst [10 * 4]uint32
	ends := [...]int{5, 8, 10}
	if err := fillGPUTree(dst[:], 10, 2, 2, ends[:]); err != nil {
		t.Fatal(err)
	}
	if allocs := testing.AllocsPerRun(1000, func() {
		if err := fillGPUTree(dst[:], 10, 2, 2, ends[:]); err != nil {
			panic(err)
		}
	}); allocs != 0 {
		t.Fatalf("warm allocs = %g", allocs)
	}
}

func TestFillGPUTreeUpperBounds(t *testing.T) {
	const (
		n    = 512
		ns   = 1
		nq   = 1
		root = ^uint32(0)
	)
	var ends [64]int
	start := ns + nq
	for i := range ends {
		step := 7
		if i < 62 {
			step = 8
		}
		start += step
		ends[i] = start
	}
	if start != n {
		t.Fatalf("bad test setup end=%d", start)
	}
	var dst [n * 4]uint32
	if err := fillGPUTree(dst[:], n, ns, nq, ends[:]); err != nil {
		t.Fatal(err)
	}
	if got := dst[:8]; !slices.Equal(got, []uint32{root, 0, 0, 1, 0, 1, 0, 2}) {
		t.Fatalf("prefix rows = %v", got)
	}
	start = ns + nq
	for i, end := range ends {
		row := dst[start*4 : start*4+4]
		want := []uint32{1, 2, uint32(start), uint32(end)}
		if !slices.Equal(row, want) {
			t.Fatalf("candidate %d start row = %v want %v", i, row, want)
		}
		start = end
	}
	if got := dst[(n-1)*4+3]; got != n {
		t.Fatalf("last end = %d", got)
	}
}
