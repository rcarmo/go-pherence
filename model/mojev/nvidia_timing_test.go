package mojev

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/loader/weights"
)

// Timing replays the final bounded group of each supplied request with events.
// It is diagnostic attribution; event overhead is not a production benchmark.
func TestMoJevNVIDIAKernelTiming(t *testing.T) {
	if os.Getenv("GO_PHERENCE_MOJEV_TIMING") != "1" {
		t.Skip("set GO_PHERENCE_MOJEV_TIMING=1 with checkpoint and workload JSON")
	}
	dir, workloads := os.Getenv("GO_PHERENCE_MOJEV_CHECKPOINT_DIR"), os.Getenv("GO_PHERENCE_MOJEV_TIMING_WORKLOADS")
	if dir == "" || workloads == "" {
		t.Fatal("checkpoint/workload paths required")
	}
	f := nativeFixture(t)
	for name, want := range map[string]string{"model.safetensors": f.WeightSHA, "config.json": f.ConfigSHA, "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523", "tokenizer_config.json": "66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b"} {
		file, e := os.Open(filepath.Join(dir, name))
		if e != nil {
			t.Fatal(e)
		}
		h := sha256.New()
		_, e = io.Copy(h, file)
		file.Close()
		if e != nil || fmt.Sprintf("%x", h.Sum(nil)) != want {
			t.Fatal("asset hash", name, e)
		}
	}
	config, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	src, err := weights.OpenSafetensors(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	cpu, err := LoadTextScorer(src, config)
	if err != nil {
		t.Fatal(err)
	}
	src.Close()
	tok, err := tokenizer.LoadWithConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	g, err := NewNVIDIATextScorer(cpu, 256)
	if err != nil {
		if g != nil {
			_ = g.Close()
		}
		t.Fatal(err)
	}
	defer func() {
		if e := g.Close(); e != nil {
			t.Error(e)
		}
	}()
	timer, err := nvidia.NewLaunchTimer(512)
	if err != nil {
		if timer != nil {
			_ = timer.Close()
		}
		t.Fatal(err)
	}
	defer func() {
		if e := timer.Close(); e != nil {
			t.Error(e)
		}
	}()
	data, err := os.ReadFile(workloads)
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name    string          `json:"name"`
		Request json.RawMessage `json:"request"`
	}
	if err = json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) < 1 || len(cases) > 16 {
		t.Fatal("bounded workload count")
	}
	names := map[nvidia.CUfunction]string{}
	for k, v := range g.kernels {
		names[v] = k
	}
	type stat struct {
		Kernel string
		Count  int
		MeanMS float64
	}
	type report struct {
		Name           string
		Rows, Launches int
		Samples        [][]float32
		Totals         []stat
	}
	var reports []report
	for _, c := range cases {
		req, e := DecodeTextRequest(bytes.NewReader(append([]byte(`{"model":"mojev",`), c.Request[1:]...)))
		if e != nil {
			t.Fatal(e)
		}
		if _, e = g.ScoreText(req, tok, 4096, 4096); e != nil {
			t.Fatal(e)
		}
		// No other caller: restore both host inputs before every replay, since the
		// preceding inference mutated x in place. Metadata belongs to this group.
		n := int(g.commands[0].Args[3])
		want := append([]float32(nil), g.hostOutput[:n*1024]...)
		r := report{Name: c.Name, Rows: n, Launches: g.commandCount}
		agg := map[string]*stat{}
		for repeat := 0; repeat < 6; repeat++ {
			if e = g.scratch["x"].Upload(g.hostInput[:n*1024]); e != nil {
				t.Fatal(e)
			}
			if e = g.tree.UploadUint32(g.treeRows[:n*4]); e != nil {
				t.Fatal(e)
			}
			ms, e := timer.Measure(g.commands[:g.commandCount])
			if e != nil {
				t.Fatal(e)
			}
			got := make([]float32, n*1024)
			if e = g.scratch["norm"].Download(got); e != nil {
				t.Fatal(e)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatal("event instrumentation changed output", c.Name, repeat)
			}
			if repeat == 0 {
				continue
			}
			r.Samples = append(r.Samples, ms)
			for i, v := range ms {
				cmd := g.commands[i]
				key := names[cmd.Function]
				if key == "mj_gemm" {
					key = fmt.Sprintf("%s/m%d/n%d/k%d", key, cmd.Args[3], cmd.Args[4], cmd.Args[5])
				}
				a := agg[key]
				if a == nil {
					a = &stat{Kernel: key}
					agg[key] = a
				}
				a.Count++
				a.MeanMS += float64(v) / 5
			}
		}
		for _, v := range agg {
			v.Count /= 5
			r.Totals = append(r.Totals, *v)
		}
		sort.Slice(r.Totals, func(i, j int) bool { return r.Totals[i].MeanMS > r.Totals[j].MeanMS })
		t.Logf("WORKLOAD %s rows=%d launches=%d", c.Name, n, g.commandCount)
		for _, v := range r.Totals {
			t.Logf("KERNEL %s count=%d mean_total_ms=%.6f", v.Kernel, v.Count, v.MeanMS)
		}
		reports = append(reports, r)
	}
	if p := os.Getenv("GO_PHERENCE_MOJEV_TIMING_REPORT"); p != "" {
		data, e := json.MarshalIndent(reports, "", "  ")
		if e != nil {
			t.Fatal(e)
		}
		if e = os.WriteFile(p, data, 0600); e != nil {
			t.Fatal(e)
		}
	}
}
