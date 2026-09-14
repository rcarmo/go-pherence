package community1

import (
	"context"
	"encoding/json"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"
)

func TestWeSpeakerTiledTiming(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_COMMUNITY1_GEMM_TIMING") != "1" {
		t.Skip("explicit CPU A/B timing opt-in")
	}
	dir, mf := trainedEmbeddingAssets(t)
	ctx := context.Background()
	src, e := safetensors.Open(filepath.Join(dir, "embedding.safetensors"))
	if e != nil {
		t.Fatal(e)
	}
	model, e := LoadWeSpeakerResNetSource(ctx, src, mf.Config, mf.Prefix)
	src.Close()
	if e != nil {
		t.Fatal(e)
	}
	oracle, e := safetensors.Open(filepath.Join(dir, "public-6-11s.safetensors"))
	if e != nil {
		t.Fatal(e)
	}
	defer oracle.Close()
	fbank := trainedTensor(t, oracle, "fbank")
	frames := len(fbank) / 80
	modes := []WeSpeakerBlockMode{WeSpeakerBlockSIMD, WeSpeakerBlockGEMM}
	want := make([][]float32, 2)
	var shape CHWShape
	for i, mode := range modes {
		want[i], shape, e = model.ForwardFrames(ctx, fbank, frames, mode)
		if e != nil {
			t.Fatal(e)
		}
	}
	if len(want[0]) != len(want[1]) {
		t.Fatal("shape")
	}
	maxerr := 0.
	for i, v := range want[0] {
		maxerr = math.Max(maxerr, math.Abs(float64(v)-float64(want[1][i])))
	}
	samples := make([][]int64, 2)
	for block := 0; block < 2; block++ {
		order := []int{0, 1, 1, 0}
		if block%2 == 1 {
			order = []int{1, 0, 0, 1}
		}
		for _, which := range order {
			start := time.Now()
			got, s, e := model.ForwardFrames(ctx, fbank, frames, modes[which])
			ns := time.Since(start).Nanoseconds()
			if e != nil || s != shape {
				t.Fatal(e)
			}
			for i, v := range got {
				if math.Float32bits(v) != math.Float32bits(want[which][i]) {
					t.Fatal("timed repeat changed", which, i)
				}
			}
			samples[which] = append(samples[which], ns)
			b, _ := json.Marshal(map[string]any{"mode": which, "block": block, "ns": ns, "repeat_bits_exact": true})
			t.Logf("TILED_SAMPLE %s", b)
		}
	}
	med := func(x []int64) float64 {
		v := append([]int64(nil), x...)
		sort.Slice(v, func(i, j int) bool { return v[i] < v[j] })
		return float64(v[1]+v[2]) / 2
	}
	a, b := med(samples[0]), med(samples[1])
	data, _ := json.Marshal(map[string]any{"baseline_median_ns": a, "candidate_median_ns": b, "speedup": a / b, "samples_per_mode": 4, "baseline_candidate_max_abs": maxerr, "scope": "resident weights+fixed reference fbank; includes packing/validation/allocations/BN; excludes model loading/frontend/pooling"})
	t.Logf("TILED_TIMING %s", data)
}
