//go:build linux

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/backends/vulkan"
)

func copyCLISoakBundle(t *testing.T, source, destination string) {
	t.Helper()
	entries, err := os.ReadDir(source)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if entry.IsDir() || entry.Type()&os.ModeSymlink != 0 || entry.Name() != "current.json" && !strings.HasSuffix(entry.Name(), ".model.bin") && !strings.HasSuffix(entry.Name(), ".run.json") {
			t.Fatal("restore entry", entry.Name())
		}
		in, err := os.Open(filepath.Join(source, entry.Name()))
		if err != nil {
			t.Fatal(err)
		}
		out, err := os.OpenFile(filepath.Join(destination, entry.Name()), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if err != nil {
			in.Close()
			t.Fatal(err)
		}
		_, err = io.Copy(out, in)
		closeOut, closeIn := out.Close(), in.Close()
		if err != nil || closeOut != nil || closeIn != nil {
			t.Fatal(err, closeOut, closeIn)
		}
	}
}

// Four NEW updates from an accepted step-two boundary; every published native
// generation gets its independent CPU counterpart. No reset/recovery/GC in the
// command, no reuse of completed qualification directories or altered gates.
func TestPocketCLIContinuationSoak(t *testing.T) {
	mode := os.Getenv("GO_PHERENCE_POCKET_CLI_SOAK_MODE")
	if mode == "" {
		t.Skip("explicit CPU/native continuation soak")
	}
	if mode != "cpu" && mode != "native" {
		t.Fatal("soak mode")
	}
	output := os.Getenv("GO_PHERENCE_POCKET_CLI_OUTPUT")
	if output == "" {
		t.Fatal("exclusive output")
	}
	run := filepath.Join(output, "run")
	if err := os.Mkdir(run, 0700); err != nil {
		t.Fatal(err)
	}
	copyCLISoakBundle(t, os.Getenv("GO_PHERENCE_POCKET_CLI_RESTORE"), run)
	var pointer struct {
		Step uint64 `json:"step"`
	}
	raw, err := os.ReadFile(filepath.Join(run, "current.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &pointer); err != nil || pointer.Step != 2 {
		t.Fatal("fresh step2 boundary", err)
	}
	backend := "simd"
	if mode == "native" {
		backend = "vulkan-resident-sharedworkspace-hybrid"
	}
	args := []string{"-mode=train", "-backend=" + backend, "-allow-experimental-sampler", "-source-revision=" + os.Getenv("GO_PHERENCE_POCKET_CLI_REVISION"), "-max-resident-bytes=6442450944", "-max-checkpoint-bytes=2147483648", "-config=../../../model/pockettts/testdata/english-upstream.yaml", "-model=" + os.Getenv("GO_PHERENCE_POCKETTTS_VOICE_MODEL"), "-model-sha256=fb0dc01b0d4d2e1c905b7a3e0676e3d9c96d5ae460e24e3ab94981805babf997", "-prepared-corpus=" + os.Getenv("GO_PHERENCE_POCKET_SHARED_CORPUS"), "-output-dir=" + run, "-steps=4", "-checkpoint-every=1", "-seed=42", "-resume"}
	if mode == "native" {
		args = append(args, "-allow-experimental-device-training", "-device-runtime-sha256="+os.Getenv("GO_PHERENCE_POCKET_CLI_RUNTIME_PIN"), "-max-device-bytes=2147483648")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 600*time.Second)
	defer cancel()
	var peak atomic.Int64
	stop, done := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.NewTicker(10 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				rss, err := trainingRSS()
				if err != nil {
					cancel()
					return
				}
				for old := peak.Load(); rss > old; old = peak.Load() {
					if peak.CompareAndSwap(old, rss) {
						break
					}
				}
			}
		}
	}()
	var report bytesBufferBounded
	err = trainPrepared(ctx, args, &report)
	close(stop)
	<-done
	t.Logf("soak mode=%s newsteps=4 commandPeakRSS=%d", mode, peak.Load())
	if err != nil {
		t.Fatal("continuation failed, no retry", err)
	}
	if peak.Load() > 6<<30 {
		t.Fatal("soak RSS gate", peak.Load())
	}
	if err = os.WriteFile(filepath.Join(output, "command.jsonl"), report.data, 0600); err != nil {
		t.Fatal(err)
	}
	stats := vulkan.VulkanMemoryStats()
	if mode == "native" && (stats.Bytes != 0 || stats.Allocations != 0 || stats.InFlight || stats.Uncertain || stats.DeviceLost || stats.PeakBytes > 2<<30 || stats.PeakAllocations > 4096) {
		t.Fatal("soak native closure", stats)
	}
	files, err := filepath.Glob(filepath.Join(run, "*.run.json"))
	if err != nil {
		t.Fatal(err)
	}
	type generation struct {
		Step       uint64 `json:"step"`
		Checkpoint struct {
			Name   string `json:"name"`
			SHA256 string `json:"sha256"`
		} `json:"checkpoint"`
	}
	generations := map[uint64]generation{}
	for _, file := range files {
		raw, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		var g generation
		if err = json.Unmarshal(raw, &g); err != nil {
			t.Fatal(err)
		}
		generations[g.Step] = g
	}
	for step := uint64(3); step <= 6; step++ {
		if _, ok := generations[step]; !ok {
			t.Fatal("missing completed generation", step)
		}
	}
	if mode == "cpu" {
		t.Logf("independent CPU continuation generations=%d", len(generations))
		return
	}
	oracle := os.Getenv("GO_PHERENCE_POCKET_CLI_SOAK_CPU")
	entries, err := filepath.Glob(filepath.Join(oracle, "*.run.json"))
	if err != nil {
		t.Fatal(err)
	}
	references := map[uint64]generation{}
	for _, file := range entries {
		raw, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		var g generation
		if err = json.Unmarshal(raw, &g); err != nil {
			t.Fatal(err)
		}
		references[g.Step] = g
	}
	for step := uint64(3); step <= 6; step++ {
		g, ref := generations[step], references[step]
		if ref.Step != step || filepath.Base(ref.Checkpoint.Name) != ref.Checkpoint.Name || len(ref.Checkpoint.SHA256) != 64 {
			t.Fatal("independent step association", step)
		}
		cpuPath := filepath.Join(oracle, ref.Checkpoint.Name)
		// Pin each independent CPU fixture by its saved immutable bundle integrity,
		// never use hash equality as a generated numerical acceptance gate.
		f, err := os.Open(cpuPath)
		if err != nil {
			t.Fatal(err)
		}
		hash := sha256.New()
		_, err = io.Copy(hash, f)
		closeErr := f.Close()
		if err != nil || closeErr != nil || hex.EncodeToString(hash.Sum(nil)) != ref.Checkpoint.SHA256 {
			t.Fatal("CPU fixture integrity", step, err, closeErr)
		}
		comparePocketCLIState(t, filepath.Join(run, g.Checkpoint.Name), cpuPath, ref.Checkpoint.SHA256, os.Getenv("GO_PHERENCE_POCKETTTS_VOICE_MODEL"), "", int(step))
		runtime.GC() // comparison-only phase after native owner closure, not command
	}
	t.Logf("six-step CLI soak native=%+v", stats)
}
