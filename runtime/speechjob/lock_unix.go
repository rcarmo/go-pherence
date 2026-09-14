//go:build linux || darwin || freebsd || openbsd || netbsd || dragonfly

package speechjob

import (
	"errors"
	"golang.org/x/sys/unix"
	"os"
)

func lockStore(f *os.File) error {
	e := unix.Flock(int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB)
	if errors.Is(e, unix.EWOULDBLOCK) || errors.Is(e, unix.EAGAIN) {
		return ErrBusy
	}
	return e
}
