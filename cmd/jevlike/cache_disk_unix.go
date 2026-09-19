//go:build linux || darwin

package main

import (
	"fmt"
	"os"
	"syscall"
)

func ensureFeatureDisk(root string, additional int64) error {
	if err := os.MkdirAll(root, 0o755); err != nil {
		return err
	}
	var s syscall.Statfs_t
	if err := syscall.Statfs(root, &s); err != nil {
		return err
	}
	free := uint64(s.Bavail) * uint64(s.Bsize)
	if additional < 0 || free < uint64(additional)+(30<<30) {
		return fmt.Errorf("cache extraction would leave less than 30 GiB free")
	}
	return nil
}
