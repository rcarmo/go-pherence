//go:build !(linux || darwin || freebsd || openbsd || netbsd || dragonfly)

package speechjob

import (
	"fmt"
	"os"
)

func lockStore(*os.File) error {
	return fmt.Errorf("speech job process locking unsupported on this platform")
}
