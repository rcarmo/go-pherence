package model

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestMTPGPUParityHonoursDisableBeforeDiscovery(t *testing.T) {
	if os.PathSeparator != '/' {
		t.Skip("fake discovery executable is a POSIX shell fixture")
	}
	// A fresh test subprocess avoids the package's lazy CUDA-init state.
	dir := t.TempDir()
	marker := filepath.Join(dir, "discovered")
	script := "#!/bin/sh\nprintf called > \"$MTP_GPU_POLICY_MARKER\"\nexit 0\n"
	if err := os.WriteFile(filepath.Join(dir, "nvidia-smi"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, value := range []string{"1", "0"} { // runtime uses any nonempty value
		cmd := exec.Command(os.Args[0], "-test.run=^TestGemma4MTPVerifierPostAttentionRMSNormGPUParity$", "-test.v")
		for _, e := range os.Environ() {
			if !strings.HasPrefix(e, "GO_PHERENCE_DISABLE_NVIDIA=") && !strings.HasPrefix(e, "PATH=") && !strings.HasPrefix(e, "MTP_GPU_POLICY_MARKER=") {
				cmd.Env = append(cmd.Env, e)
			}
		}
		cmd.Env = append(cmd.Env, "GO_PHERENCE_DISABLE_NVIDIA="+value, "PATH="+dir, "MTP_GPU_POLICY_MARKER="+marker)
		out, err := cmd.CombinedOutput()
		if err != nil || !strings.Contains(string(out), "--- SKIP: TestGemma4MTPVerifierPostAttentionRMSNormGPUParity") {
			t.Fatalf("disable=%q: %v\n%s", value, err, out)
		}
		if _, err := os.Stat(marker); !os.IsNotExist(err) {
			t.Fatalf("disabled test performed GPU discovery: %v", err)
		}
	}
}
