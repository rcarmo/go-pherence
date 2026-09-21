//go:build !linux

package commandcapture

import "os/exec"

func configureOwnedCommand(*exec.Cmd) {}
