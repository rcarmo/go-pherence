package main

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func TestProfilesNoOverwrite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "keep")
	if err := os.WriteFile(path, []byte("keep"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, pair := range [][2]string{{path, ""}, {"", path}, {path, path}} {
		if _, err := startProfiles(pair[0], pair[1]); err == nil {
			t.Fatal("existing profile overwritten")
		}
	}
	got, _ := os.ReadFile(path)
	if string(got) != "keep" {
		t.Fatal("changed file")
	}
}
func TestProfileFilesCreated(t *testing.T) {
	// CPU profiling is process-global. The parent test harness already owns
	// it; exercise command profiling in a separately profiled child, retaining
	// its own binary/log/profiles rather than skipping this contract.
	if dir := os.Getenv("GO_PHERENCE_OMNIVOICE_PROFILE_CHILD"); dir != "" {
		stop, err := startProfiles(filepath.Join(dir, "cpu.pprof"), filepath.Join(dir, "heap.pprof"))
		if err != nil {
			t.Fatal(err)
		}
		values := make([]byte, 1<<20)
		for i := range values {
			values[i] = byte(i)
		}
		deadline := time.Now().Add(250 * time.Millisecond)
		var sum uint64
		for time.Now().Before(deadline) {
			for _, value := range values {
				sum += uint64(value)
			}
		}
		runtime.KeepAlive(sum)
		runtime.KeepAlive(values)
		stop()
		return
	}
	root := os.Getenv("PROFILE_EVIDENCE_ROOT")
	if root == "" {
		t.Fatal("set PROFILE_EVIDENCE_ROOT to retain command-profile child evidence")
	}
	dir, err := os.MkdirTemp(root, "omnivoice-profile-child-")
	if err != nil {
		t.Fatal(err)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(dir, "package-test")
	in, err := os.Open(executable)
	if err != nil {
		t.Fatal(err)
	}
	out, err := os.OpenFile(binary, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0700)
	if err != nil {
		in.Close()
		t.Fatal(err)
	}
	_, copyErr := io.Copy(out, in)
	closeOut, closeIn := out.Close(), in.Close()
	if copyErr != nil || closeOut != nil || closeIn != nil {
		t.Fatal(copyErr, closeOut, closeIn)
	}
	receipt := fmt.Sprintf("toolchain=%s\ncommand=%s -test.run=^TestProfileFilesCreated$ -test.count=1 -test.v\nCPU=default 100Hz; memprofilerate=1\nworkload=1MiB byte loop for250ms, profile creation/finalization\n", runtime.Version(), binary)
	if err = os.WriteFile(filepath.Join(dir, "environment.txt"), []byte(receipt), 0600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(binary, "-test.run=^TestProfileFilesCreated$", "-test.count=1", "-test.v")
	cmd.Env = append(os.Environ(), "GO_PHERENCE_OMNIVOICE_PROFILE_CHILD="+dir)
	log, runErr := cmd.CombinedOutput()
	if err = os.WriteFile(filepath.Join(dir, "run.log"), log, 0600); err != nil {
		t.Fatal(err)
	}
	if runErr != nil {
		t.Fatalf("profile child failed: %v: %s", runErr, log)
	}
	for _, name := range []string{"cpu.pprof", "heap.pprof"} {
		stat, err := os.Stat(filepath.Join(dir, name))
		if err != nil || stat.Size() == 0 {
			t.Fatalf("profile %s missing", name)
		}
	}
	for _, sample := range []string{"cpu", "alloc_space", "alloc_objects"} {
		args := []string{"tool", "pprof", "-top", "-cum"}
		profile := "cpu.pprof"
		if sample != "cpu" {
			args = append(args, "-sample_index="+sample)
			profile = "heap.pprof"
		}
		args = append(args, binary, filepath.Join(dir, profile))
		out, err := exec.Command("go", args...).CombinedOutput()
		if err != nil {
			t.Fatalf("child profile analysis %s: %v: %s", sample, err, out)
		}
		if err = os.WriteFile(filepath.Join(dir, sample+".txt"), out, 0600); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("retained command profile child evidence: %s", dir)
}
