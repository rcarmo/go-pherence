//go:build linux

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/model/pockettts"
)

// Actual command path, not the lower-level integrated runner. Output directory
// must be fresh; failure never publishes a retry or relaxes RSS/native gates.
func TestPocketTrainSharedReleasedCommand(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_CLI_SHARED_NATIVE") != "1" {
		t.Skip("explicit released CLI command qualification")
	}
	output := os.Getenv("GO_PHERENCE_POCKET_CLI_OUTPUT")
	root := filepath.Join(output, "run")
	if output == "" {
		t.Fatal("fresh evidence root required")
	}
	if err := os.Mkdir(root, 0700); err != nil {
		t.Fatal(err)
	}
	expectedStep := 1
	cpuHash := "d0d366f789a6ab0078bd0f1534d16a38dddb4e08a3193797be23c9f6b1794e5c"
	if restore := os.Getenv("GO_PHERENCE_POCKET_CLI_RESTORE"); restore != "" {
		files, err := os.ReadDir(restore)
		if err != nil {
			t.Fatal(err)
		}
		for _, entry := range files {
			if entry.IsDir() || entry.Type()&os.ModeSymlink != 0 {
				t.Fatal("restore bundle entry", entry.Name())
			}
			if entry.Name() != "current.json" && !strings.HasSuffix(entry.Name(), ".model.bin") && !strings.HasSuffix(entry.Name(), ".run.json") {
				t.Fatal("unexpected restore entry", entry.Name())
			}
			in, err := os.Open(filepath.Join(restore, entry.Name()))
			if err != nil {
				t.Fatal(err)
			}
			out, err := os.OpenFile(filepath.Join(root, entry.Name()), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
			if err != nil {
				in.Close()
				t.Fatal(err)
			}
			_, copyErr := io.Copy(out, in)
			closeOut, closeIn := out.Close(), in.Close()
			if copyErr != nil || closeOut != nil || closeIn != nil {
				t.Fatal(copyErr, closeOut, closeIn)
			}
		}
		expectedStep = 2
		cpuHash = "16391d753802996f7fded00af5f5c8391d699c69c48aa2563cc77d3659de7413"
	}
	model := os.Getenv("GO_PHERENCE_POCKETTTS_VOICE_MODEL")
	cpu := os.Getenv("GO_PHERENCE_POCKET_CLI_CPU")
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	var report bytesBufferBounded
	args := []string{"-mode=train", "-backend=vulkan-resident-sharedworkspace-hybrid", "-allow-experimental-device-training", "-allow-experimental-sampler", "-source-revision=" + os.Getenv("GO_PHERENCE_POCKET_CLI_REVISION"), "-device-runtime-sha256=" + os.Getenv("GO_PHERENCE_POCKET_CLI_RUNTIME_PIN"), "-max-device-bytes=2147483648", "-max-resident-bytes=6442450944", "-config=../../../model/pockettts/testdata/english-upstream.yaml", "-model=" + model, "-model-sha256=fb0dc01b0d4d2e1c905b7a3e0676e3d9c96d5ae460e24e3ab94981805babf997", "-prepared-corpus=" + os.Getenv("GO_PHERENCE_POCKET_SHARED_CORPUS"), "-output-dir=" + root, "-steps=1", "-checkpoint-every=1", "-seed=42"}
	exportPath := filepath.Join(output, "ema.safetensors")
	if expectedStep == 2 {
		args = append(args, "-resume", "-export="+exportPath)
	}
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
	commandErr := trainPrepared(ctx, args, &report)
	close(stop)
	<-done
	t.Logf("command-only step%d peakRSS=%d", expectedStep, peak.Load())
	if commandErr != nil {
		t.Fatal("released CLI failed, no retry", commandErr)
	}
	if peak.Load() > 6<<30 {
		t.Fatal("command RSS exceeds gate", peak.Load())
	}
	if err := os.WriteFile(filepath.Join(output, "command.jsonl"), report.data, 0600); err != nil {
		t.Fatal(err)
	}
	stats := vulkan.VulkanMemoryStats()
	if stats.Bytes != 0 || stats.Allocations != 0 || stats.InFlight || stats.Uncertain || stats.DeviceLost || stats.PeakBytes > 2<<30 || stats.PeakAllocations > 4096 {
		t.Fatal("CLI native closure", stats)
	}
	matches, err := filepath.Glob(filepath.Join(root, "*.model.bin"))
	if err != nil || len(matches) != expectedStep {
		t.Fatal("command checkpoint", matches, err)
	}
	var pointer struct {
		ModelName string `json:"model_name"`
	}
	pointerRaw, err := os.ReadFile(filepath.Join(root, "current.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(pointerRaw, &pointer); err != nil || filepath.Base(pointer.ModelName) != pointer.ModelName {
		t.Fatal("checkpoint pointer", err)
	}
	current := filepath.Join(root, pointer.ModelName)
	comparePocketCLIState(t, current, cpu, cpuHash, model, exportPath, expectedStep)
	t.Logf("command native=%+v checkpoint=%s", stats, current)
}

func TestPocketTrainSharedSavedComparison(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_CLI_SAVED_COMPARE") != "1" {
		t.Skip("explicit saved CLI result comparison; no native run")
	}
	root := os.Getenv("GO_PHERENCE_POCKET_CLI_SAVED_RESULT")
	raw, err := os.ReadFile(filepath.Join(root, "run/current.json"))
	if err != nil {
		t.Fatal(err)
	}
	var pointer struct {
		ModelName string `json:"model_name"`
	}
	if err = json.Unmarshal(raw, &pointer); err != nil || filepath.Base(pointer.ModelName) != pointer.ModelName {
		t.Fatal("saved pointer", err)
	}
	comparePocketCLIState(t, filepath.Join(root, "run", pointer.ModelName), os.Getenv("GO_PHERENCE_POCKET_CLI_CPU"), "16391d753802996f7fded00af5f5c8391d699c69c48aa2563cc77d3659de7413", os.Getenv("GO_PHERENCE_POCKETTTS_VOICE_MODEL"), filepath.Join(root, "ema.safetensors"), 2)
}

func comparePocketCLIState(t *testing.T, current, cpu, cpuHash, model, exportPath string, expectedStep int) {
	t.Helper()
	file, err := os.Open(cpu)
	if err != nil {
		t.Fatal(err)
	}
	hash := sha256.New()
	n, err := io.Copy(hash, file)
	closeErr := file.Close()
	if err != nil || closeErr != nil || n != 1431217112 || hex.EncodeToString(hash.Sum(nil)) != cpuHash {
		t.Fatal("independent CPU pin", err, closeErr)
	}
	// Coarse diagnostic phase after command ownership closure, not production
	// GC or admission policy. Independent comparison storage is a separate phase.
	runtime.GC()
	native, err := pockettts.LoadFullTrainingStateBinary(current, 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	reference, err := pockettts.LoadFullTrainingStateBinary(cpu, 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	compare := func(label string, got, want []pockettts.NamedTrainingTensor) {
		if len(got) != len(want) {
			t.Fatal("inventory", label)
		}
		var maxError float64
		var count, above int64
		for i := range got {
			if got[i].Name != want[i].Name || len(got[i].Values) != len(want[i].Values) {
				t.Fatal("tensor association", label)
			}
			for j, x := range got[i].Values {
				d := math.Abs(float64(x) - float64(want[i].Values[j]))
				if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
					t.Fatal("nonfinite native")
				}
				maxError = max(maxError, d)
				if d > 4e-6 {
					above++
				}
				count++
			}
		}
		t.Logf("CLI %s values=%d max=%g above4e6=%d", label, count, maxError, above)
		if above != 0 {
			t.Fatal("independent gate", label, above)
		}
	}
	if native.Step != expectedStep || reference.Step != expectedStep {
		t.Fatal("step association")
	}
	compare("parameters", native.Params, reference.Params)
	compare("EMA", native.EMA, reference.EMA)
	if expectedStep == 2 {
		frozen, err := safetensors.Open(model)
		if err != nil {
			t.Fatal(err)
		}
		defer frozen.Close()
		exported, err := safetensors.Open(exportPath)
		if err != nil {
			t.Fatal(err)
		}
		defer exported.Close()
		values := map[string][]float32{}
		for _, tensor := range reference.EMA {
			values[tensor.Name] = tensor.Values
		}
		for _, tensor := range reference.Buffers {
			values[tensor.Name] = tensor.Values
		}
		var count int64
		var maxError float64
		for _, name := range exported.Names() {
			got, _, err := exported.GetFloat32(name)
			if err != nil {
				t.Fatal(err)
			}
			want := values[name]
			if strings.HasPrefix(name, "mimi.") {
				want, _, err = frozen.GetFloat32(name)
				if err != nil {
					t.Fatal(err)
				}
			}
			if len(got) != len(want) {
				t.Fatal("export inventory", name)
			}
			for i, x := range got {
				d := math.Abs(float64(x) - float64(want[i]))
				if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || d > 6e-8 {
					t.Fatal("independent export gate", name, i, d)
				}
				maxError = max(maxError, d)
				count++
			}
		}
		t.Logf("CLI independent EMA export tensors=%d values=%d max=%g gate6e-8", len(exported.Names()), count, maxError)
	}
}

type bytesBufferBounded struct{ data []byte }

func (b *bytesBufferBounded) Write(data []byte) (int, error) {
	if len(b.data)+len(data) > 1<<20 {
		return 0, fmt.Errorf("command report ceiling")
	}
	b.data = append(b.data, data...)
	return len(data), nil
}
