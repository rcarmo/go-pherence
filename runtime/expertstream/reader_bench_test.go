package expertstream

import "testing"

func benchmarkReader(b *testing.B, expertBytes, slots, workers int) (*Reader, []uint64) {
	b.Helper()
	dir := b.TempDir()
	defs := make([]expertDef, slots)
	keys := make([]uint64, slots)
	// Three equal components make the total payload approximately expertBytes.
	componentFloats := int64(expertBytes / (3 * 4))
	if componentFloats < 1 {
		componentFloats = 1
	}
	for i := range defs {
		keys[i] = uint64(i + 1)
		defs[i] = expertDef{key: keys[i], gateN: componentFloats, upN: componentFloats, downN: componentFloats}
	}
	manifest, _, _ := buildFixture(b, dir, 4096, defs)
	path := writeManifest(b, dir, manifest)
	r, err := Open(path, Options{Slots: slots, Workers: workers})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = r.Close() })
	return r, keys
}

func BenchmarkReaderLoadWarmHit1MiB(b *testing.B) {
	r, keys := benchmarkReader(b, 1<<20, 4, 2)
	if _, err := r.Load(keys); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := r.Load(keys); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkReaderLoadColdMiss1MiB(b *testing.B) {
	// Alternate two disjoint four-expert sets through four slots, forcing every
	// measured load to issue four bounded ReadAt misses without relying on OS
	// page-cache eviction (this is a reader cold miss, not a disk-cold claim).
	r, keys := benchmarkReader(b, 1<<20, 8, 4)
	r.slots = r.slots[:4]
	first, second := keys[:4], keys[4:]
	if _, err := r.Load(first); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(4 * (1 << 20))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		set := second
		if i&1 != 0 {
			set = first
		}
		if _, err := r.Load(set); err != nil {
			b.Fatal(err)
		}
	}
}
