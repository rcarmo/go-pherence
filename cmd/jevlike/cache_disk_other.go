//go:build !linux && !darwin

package main

import "fmt"

func ensureFeatureDisk(root string, additional int64) error {
	return fmt.Errorf("feature extraction disk budget check unsupported on this OS")
}
