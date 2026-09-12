//go:build linux

package media

import (
	"os"
	"os/exec"
	"syscall"
)

// configureOwnedCommand isolates the media subprocess in its own process group.
// os/exec invokes Cancel when the context expires; signalling -pid terminates the
// complete owned group, including any descendants. Pdeathsig covers abrupt Go
// process death before it can run cancellation cleanup.
func configureOwnedCommand(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true, Pdeathsig: syscall.SIGKILL}
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return os.ErrProcessDone
		}
		err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		if err == syscall.ESRCH {
			return os.ErrProcessDone
		}
		return err
	}
}
