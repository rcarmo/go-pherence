//go:build !linux || !riscv64

package tcm

import "testing"

func TestUnsupportedHost(t *testing.T) {
	if IsAvailable() {
		t.Fatal("TCM must not be available on a foreign host")
	}
	if device, err := Open(); err == nil || device != nil {
		t.Fatalf("Open() = %v, %v; want nil and unsupported error", device, err)
	}
	var device TCM
	if ptr, err := device.Get(0); ptr != nil || err == nil {
		t.Fatalf("Get() = %v, %v", ptr, err)
	}
	if device.Acquire(0) == nil || device.Ptr(0) != nil || device.Slice(0) != nil {
		t.Fatal("stub exposed device memory")
	}
	device.Release(0)
	if err := device.Close(); err != nil {
		t.Fatal(err)
	}
}
