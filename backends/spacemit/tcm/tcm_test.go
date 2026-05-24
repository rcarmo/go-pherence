package tcm

import (
	"testing"
	"unsafe"
)

func TestOpen(t *testing.T) {
	if !IsAvailable() {
		t.Skip("TCM not available")
	}
	tcm, err := Open()
	if err != nil {
		t.Fatal(err)
	}
	defer tcm.Close()

	t.Logf("TCM opened: base=%#x", tcm.base)

	// Test Get/Release
	ptr, err := tcm.Get(0)
	if err != nil {
		t.Fatal(err)
	}
	if ptr == nil {
		t.Fatal("Get returned nil")
	}
	t.Logf("Block 0 at %p", ptr)

	// Write test pattern
	buf := (*[16]byte)(ptr)
	for i := range buf {
		buf[i] = byte(i + 0xA0)
	}

	// Read back
	for i := range buf {
		if buf[i] != byte(i+0xA0) {
			t.Errorf("readback[%d]: got %#x want %#x", i, buf[i], byte(i+0xA0))
		}
	}

	tcm.Release(0)
	t.Log("Write/read/release OK")
}

func TestAllBlocks(t *testing.T) {
	if !IsAvailable() {
		t.Skip("TCM not available")
	}
	tcm, err := Open()
	if err != nil {
		t.Fatal(err)
	}
	defer tcm.Close()

	// Write unique pattern to each block, verify isolation
	for i := 0; i < BlockCount; i++ {
		s := tcm.Slice(i)
		s[0] = byte(i + 0x50)
		s[1] = byte(i + 0x50)
		s[BlockSize-1] = byte(i + 0x60)
	}

	for i := 0; i < BlockCount; i++ {
		s := tcm.Slice(i)
		if s[0] != byte(i+0x50) {
			t.Errorf("block %d start: got %#x want %#x", i, s[0], byte(i+0x50))
		}
		if s[BlockSize-1] != byte(i+0x60) {
			t.Errorf("block %d end: got %#x want %#x", i, s[BlockSize-1], byte(i+0x60))
		}
	}
	t.Log("All 8 blocks independently addressable")
}

func BenchmarkTCMWrite(b *testing.B) {
	if !IsAvailable() {
		b.Skip("TCM not available")
	}
	tcm, err := Open()
	if err != nil {
		b.Fatal(err)
	}
	defer tcm.Close()

	src := make([]byte, 32*1024) // 32KB chunk
	for i := range src {
		src[i] = byte(i)
	}
	dst := tcm.Slice(0)

	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(dst, src)
	}
}

func BenchmarkTCMRead(b *testing.B) {
	if !IsAvailable() {
		b.Skip("TCM not available")
	}
	tcm, err := Open()
	if err != nil {
		b.Fatal(err)
	}
	defer tcm.Close()

	dst := make([]byte, 32*1024)
	src := tcm.Slice(0)
	// Pre-fill
	for i := range src[:len(dst)] {
		src[i] = byte(i)
	}

	b.SetBytes(int64(len(dst)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(dst, src[:len(dst)])
	}
}

func BenchmarkDRAMWrite(b *testing.B) {
	src := make([]byte, 32*1024)
	dst := make([]byte, 384*1024) // same size as TCM block
	for i := range src {
		src[i] = byte(i)
	}

	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(dst, src)
	}
}

// Ensure unsafe.Pointer usage compiles
var _ = unsafe.Pointer(nil)
