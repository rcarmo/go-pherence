package kv

import "testing"

func compressedFixture(t *testing.T) *CompressedKVCache {
	t.Helper()
	cfg := DefaultTurboQuantConfig()
	cfg.ResidualWindow = 0
	c := NewCompressedKVCache(4, 1, 4, NewTurboQuantState(4, 1, cfg), false)
	c.Append([]float32{1, 2, 3, 4}, []float32{4, 3, 2, 1})
	if len(c.CompressedK) != 1 || len(c.CompressedV) != 1 {
		t.Fatal("fixture did not compress")
	}
	return c
}
func TestCompressedCacheRejectsInconsistentExportedStorage(t *testing.T) {
	for _, values := range []bool{false, true} {
		c := compressedFixture(t)
		if values {
			c.CompressedV = append(c.CompressedV, c.CompressedV[0])
			if got := c.GetV(); len(got) != len(c.FullV) {
				t.Fatal(got)
			}
		} else {
			c.CompressedK = append(c.CompressedK, c.CompressedK[0])
			if got := c.GetK(); len(got) != len(c.FullK) {
				t.Fatal(got)
			}
		}
	}
	c := NewCompressedKVCache(4, 1, 4, NewTurboQuantState(4, 1, DefaultTurboQuantConfig()), false)
	c.Append([]float32{1, 2, 3, 4}, []float32{4, 3, 2, 1})
	c.FullV = nil
	kLen, seq := len(c.FullK), c.SeqLen()
	c.Append([]float32{1, 2, 3, 4}, []float32{4, 3, 2, 1})
	if len(c.FullK) != kLen || len(c.FullV) != 0 || c.SeqLen() != seq {
		t.Fatal("append mutated inconsistent storage")
	}
}
func TestCompressedCacheResetClearsRetainedEntries(t *testing.T) {
	c := compressedFixture(t)
	ks, vs := c.CompressedK, c.CompressedV
	c.Reset()
	for _, entries := range [][]compressedEntry{ks, vs} {
		for _, entry := range entries {
			if entry.Packed != nil || entry.HeadVMin != nil || entry.HeadScale != nil {
				t.Fatal("reset retained compressed payload")
			}
		}
	}
	c.Append([]float32{1, 2, 3, 4}, []float32{4, 3, 2, 1})
	if c.SeqLen() != 1 || len(c.GetK()) != 4 || len(c.GetV()) != 4 {
		t.Fatal("reset reuse failed")
	}
}
func TestCompressedCacheRejectsWrappedHeadGeometry(t *testing.T) {
	c := NewCompressedKVCache(4, int(^uint(0)>>1)/2+2, 4, nil, false)
	if c.numKVHeads != 0 || c.headDim != 0 {
		t.Fatal("wrapped head product accepted")
	}
}
