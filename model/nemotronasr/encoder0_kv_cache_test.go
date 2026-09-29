package nemotronasr

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedEncoder0KVCachePyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m, loadErr := LoadEncoder0QKV(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	input := readStemFixture(t, "encoder0_ff1_residual", 5*encoderWidth)
	kRow, vRow := make([]float32, 5*encoderWidth), make([]float32, 5*encoderWidth)
	_, k, v, err := m.Project(input, 5)
	if err != nil {
		t.Fatal(err)
	}
	copy(kRow, k)
	copy(vRow, v)
	var cache Encoder0KVCache
	for index, bounds := range [][2]int{{0, 1}, {1, 3}, {3, 5}} {
		frames := bounds[1] - bounds[0]
		chunkK, chunkV := make([]float32, encoderWidth*frames), make([]float32, encoderWidth*frames)
		for head := 0; head < asrAttentionHeads; head++ {
			for row := 0; row < frames; row++ {
				for dim := 0; dim < asrAttentionHeadWidth; dim++ {
					from := (bounds[0]+row)*encoderWidth + head*asrAttentionHeadWidth + dim
					to := head*frames*asrAttentionHeadWidth + row*asrAttentionHeadWidth + dim
					chunkK[to], chunkV[to] = kRow[from], vRow[from]
				}
			}
		}
		originalK := append([]float32(nil), chunkK...)
		originalV := append([]float32(nil), chunkV...)
		visibleK, visibleV, err := cache.Update(chunkK, chunkV, frames)
		if err != nil {
			t.Fatal(err)
		}
		stateK, stateV, retained, seen := cache.Snapshot()
		if retained != bounds[1] || seen != bounds[1] {
			t.Fatalf("chunk=%d retained=%d seen=%d", index, retained, seen)
		}
		for _, item := range []struct {
			name string
			got  []float32
		}{{"returned_k", visibleK}, {"returned_v", visibleV}, {"state_k", stateK}, {"state_v", stateV}} {
			ref := readStemFixture(t, fmt.Sprintf("encoder0_attn_cache_%s%d", item.name, index), len(item.got))
			var maxAbs, sumAbs float64
			var outside int
			for i, value := range item.got {
				delta := math.Abs(float64(value - ref[i]))
				maxAbs = math.Max(maxAbs, delta)
				sumAbs += delta
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(ref[i])) {
					outside++
				}
			}
			mean := sumAbs / float64(len(item.got))
			t.Logf("chunk=%d %s max_abs=%g mean_abs=%g outside=%d", index, item.name, maxAbs, mean, outside)
			if outside != 0 || mean > 2e-5 {
				t.Fatalf("chunk=%d %s differs from independent reference", index, item.name)
			}
		}
		for i, value := range chunkK {
			if value != originalK[i] || chunkV[i] != originalV[i] {
				t.Fatalf("mutated caller key/value at %d", i)
			}
		}
		last := len(stateK) - 1
		before := stateK[last]
		visibleK[len(visibleK)-1]++
		stateK[last]++
		stateV[last]++
		visibleV[len(visibleV)-1]++
		stateCheck, valueCheck, _, _ := cache.Snapshot()
		if stateCheck[last] != before || valueCheck[last] != originalV[len(originalV)-1] {
			t.Fatal("returned/snapshot K/V aliases cache")
		}
	}
}

func TestEncoder0KVCacheOverflowAndErrors(t *testing.T) {
	var cache Encoder0KVCache
	for _, tc := range []struct{ first, frames int }{{0, 40}, {40, 20}, {60, 7}} {
		k, v := make([]float32, encoderWidth*tc.frames), make([]float32, encoderWidth*tc.frames)
		for head := 0; head < asrAttentionHeads; head++ {
			for row := 0; row < tc.frames; row++ {
				for dim := 0; dim < asrAttentionHeadWidth; dim++ {
					i := head*tc.frames*asrAttentionHeadWidth + row*asrAttentionHeadWidth + dim
					k[i], v[i] = float32(tc.first+row), -float32(tc.first+row)
				}
			}
		}
		visibleK, _, err := cache.Update(k, v, tc.frames)
		if err != nil {
			t.Fatal(err)
		}
		state, _, retained, seen := cache.Snapshot()
		wantVisible := tc.first + tc.frames
		if tc.first == 60 {
			wantVisible = 63
		}
		if len(visibleK) != encoderWidth*wantVisible || retained != min(tc.first+tc.frames, 56) || seen != tc.first+tc.frames {
			t.Fatalf("first=%d visible=%d retained=%d seen=%d", tc.first, len(visibleK)/encoderWidth, retained, seen)
		}
		wantFirst := max(0, seen-56)
		if state[0] != float32(wantFirst) || visibleK[0] != float32(max(0, tc.first-56)) {
			t.Fatalf("first=%d state0=%g visible0=%g", tc.first, state[0], visibleK[0])
		}
	}
	beforeK, beforeV, retained, seen := cache.Snapshot()
	for _, bad := range []struct {
		keys, values []float32
		frames       int
	}{
		{nil, nil, 0}, {make([]float32, encoderWidth-1), make([]float32, encoderWidth), 1},
		{make([]float32, encoderWidth), make([]float32, encoderWidth-1), 1},
		{[]float32{float32(math.NaN())}, nil, 1},
	} {
		if _, _, err := cache.Update(bad.keys, bad.values, bad.frames); err == nil {
			t.Fatal("accepted malformed KV update")
		}
	}
	afterK, afterV, afterRetained, afterSeen := cache.Snapshot()
	if retained != afterRetained || seen != afterSeen || len(beforeK) != len(afterK) || len(beforeV) != len(afterV) {
		t.Fatal("mutated cache metadata on error")
	}
	for i := range beforeK {
		if beforeK[i] != afterK[i] || beforeV[i] != afterV[i] {
			t.Fatalf("mutated cache on error at %d", i)
		}
	}
}
