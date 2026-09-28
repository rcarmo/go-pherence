package needle

import (
	"crypto/sha256"
	"fmt"
	"os"
	"sort"
	"testing"
)

func TestReleasedPackedArchiveBitInventory(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEEDLE_PACKED_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEEDLE_PACKED_MODEL to the pinned local archive")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != "c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38" {
		t.Fatalf("unexpected archive sha256 %s", got)
	}
	a, err := ParseArchivePacked(data)
	if err != nil {
		t.Fatal(err)
	}
	counts := map[int]int{}
	for _, record := range a.Records {
		if record.DType == archiveDTypeCQ {
			counts[record.Bits]++
		}
	}
	bits := make([]int, 0, len(counts))
	for bit := range counts {
		bits = append(bits, bit)
	}
	sort.Ints(bits)
	for _, bit := range bits {
		t.Logf("released CQ%d records=%d", bit, counts[bit])
	}
}
