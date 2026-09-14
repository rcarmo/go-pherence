//go:build !linux

package media

import "os/exec"

func configureOwnedCommand(*exec.Cmd) {}
