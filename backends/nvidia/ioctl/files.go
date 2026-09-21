package nv

import (
	"fmt"
	"golang.org/x/sys/unix"
)

// The injected descriptor functions let host tests prove cleanup without ever
// opening an NVIDIA device or issuing an ioctl. A zero-value device owns no fds.
func (d *NVDevice) openDeviceFiles(open func(string, int, uint32) (int, error), close func(int) error) error {
	if d == nil || d.filesOpened {
		return fmt.Errorf("invalid/reopened NV device")
	}
	d.fdCtl, d.fdDev, d.fdDevAlloc, d.fdUVM, d.fdUVM2 = -1, -1, -1, -1, -1
	d.filesOpened = true
	for _, entry := range []struct {
		path string
		dst  *int
	}{{"/dev/nvidiactl", &d.fdCtl}, {"/dev/nvidia0", &d.fdDev}, {"/dev/nvidia0", &d.fdDevAlloc}, {"/dev/nvidia-uvm", &d.fdUVM}, {"/dev/nvidia-uvm", &d.fdUVM2}} {
		fd, err := open(entry.path, unix.O_RDWR|unix.O_CLOEXEC, 0)
		if err != nil {
			d.closeDeviceFiles(close)
			return fmt.Errorf("open %s: %w", entry.path, err)
		}
		*entry.dst = fd
	}
	return nil
}
func (d *NVDevice) closeDeviceFiles(close func(int) error) {
	if d == nil || !d.filesOpened {
		return
	}
	// Release secondary descriptors before the root control descriptor.
	for _, fd := range []*int{&d.fdUVM2, &d.fdUVM, &d.fdDevAlloc, &d.fdDev, &d.fdCtl} {
		if *fd >= 0 {
			_ = close(*fd)
			*fd = -1
		}
	}
	d.filesOpened = false
}
