package audio

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestWhisperLogMel128TransformersFixture(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "whisper_logmel128_transformers_4_57_1.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Cases []struct {
			Name    string    `json:"name"`
			Samples []float32 `json:"samples"`
			Shape   []int     `json:"shape"`
			Mel     []float32 `json:"mel"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) != 4 {
		t.Fatal("missing reference cases")
	}
	for _, c := range fixture.Cases {
		t.Run(c.Name, func(t *testing.T) {
			got, frames, err := WhisperLogMel(c.Samples, 128)
			if err != nil {
				t.Fatal(err)
			}
			if len(c.Shape) != 2 || c.Shape[0] != 128 || frames != c.Shape[1] || len(got) != len(c.Mel) {
				t.Fatalf("unexpected shape128x%d len%d", frames, len(got))
			}
			for i, v := range got {
				if diff := math.Abs(float64(v - c.Mel[i])); diff > 1e-5 {
					t.Fatalf("[%d]=%.9g want%.9g diff%.9g", i, v, c.Mel[i], diff)
				}
			}
		})
	}
}

func TestWhisperLogMelCheckedInputAndBoundaries(t *testing.T) {
	for _, tt := range []struct {
		name    string
		samples []float32
		bands   int
	}{
		{"empty", nil, 128}, {"no-frame", make([]float32, 159), 128}, {"oversize", make([]float32, 480001), 128},
		{"unsupported", make([]float32, 160), 64}, {"nan", append([]float32{float32(math.NaN())}, make([]float32, 159)...), 128},
		{"infinite", append([]float32{float32(math.Inf(1))}, make([]float32, 159)...), 80},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, n, err := WhisperLogMel(tt.samples, tt.bands)
			if err == nil || got != nil || n != 0 {
				t.Fatal("accepted invalid input")
			}
		})
	}
	for _, bands := range []int{80, 128} {
		got, n, err := WhisperLogMel(make([]float32, 480000), bands)
		if err != nil || n != 3000 || len(got) != bands*3000 {
			t.Fatal("bad full-window silence")
		}
		for _, v := range got {
			if v != -1.5 {
				t.Fatal("bad silence value")
			}
		}
	}
}

func TestWhisperLogMelTablesConcurrent(t *testing.T) {
	samples := make([]float32, 320)
	samples[10] = 0.5
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func(bands int) {
			defer wg.Done()
			got, n, err := WhisperLogMel(samples, bands)
			if err != nil || n != 2 || len(got) != bands*n {
				t.Errorf("bad concurrent result: %v", err)
			}
		}(80 + 48*(i%2))
	}
	wg.Wait()
}
