package testexec

import (
	"os"
	"os/exec"
	"testing"
)

// HelperProcess lets command tests re-exec their own test binary and dispatch
// to main after a "--" separator when GO_WANT_HELPER_PROCESS=1 is set.
func HelperProcess(t *testing.T, main func()) {
	t.Helper()
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	for i, arg := range os.Args {
		if arg == "--" {
			os.Args = append([]string{os.Args[0]}, os.Args[i+1:]...)
			main()
			return
		}
	}
	os.Exit(2)
}

// RunInspectRaw re-execs the test binary as the inspected command (via the
// TestHelperProcess dispatcher) with the given args and returns its combined
// output.
func RunInspectRaw(args ...string) (string, error) {
	cmd := exec.Command(os.Args[0], append([]string{"-test.run=TestHelperProcess", "--"}, args...)...)
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
	out, err := cmd.CombinedOutput()
	return string(out), err
}

// RunInspect is RunInspectRaw but fails the test on a non-zero exit.
func RunInspect(t *testing.T, args ...string) string {
	t.Helper()
	out, err := RunInspectRaw(args...)
	if err != nil {
		t.Fatalf("inspect failed: %v\n%s", err, out)
	}
	return out
}

// WriteFile writes data to path, failing the test on error.
func WriteFile(t *testing.T, path, data string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(data), 0644); err != nil {
		t.Fatal(err)
	}
}
