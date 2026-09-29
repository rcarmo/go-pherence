package nemotronasr

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestCachedEncoderTowerRejectsMalformed(t *testing.T) {
	if _, err := (*CachedEncoderTower)(nil).ForwardChunk(make([]float32, encoderWidth), 1, 0); err == nil {
		t.Fatal("accepted nil stream")
	}
	s := &CachedEncoderTower{Tower: &OfflineEncoderTower{}}
	if _, err := s.ForwardChunk(make([]float32, encoderWidth), 1, 0); err == nil || s.seen != 0 {
		t.Fatal("accepted missing layer or advanced cache")
	}
	for _, rows := range []int{0, 6} {
		if _, err := s.ForwardChunk(make([]float32, rows*encoderWidth), rows, 0); err == nil {
			t.Fatal("accepted invalid rows")
		}
	}
}

func TestReleasedCachedEncoderTowerPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	refDir := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_TOWER_CACHED_REF")
	if path == "" || refDir == "" {
		t.Skip("set pinned model and GO_PHERENCE_NEMOTRON_ASR_TOWER_CACHED_REF")
	}
	rows := 8
	if setting := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_TOWER_CACHED_ROWS"); setting != "" {
		var err error
		rows, err = strconv.Atoi(setting)
		if err != nil || (rows != 8 && rows != 72) {
			t.Fatal("invalid cached tower reference rows")
		}
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	tower, loadErr := LoadOfflineEncoderTower(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	read := func(name string) []float32 {
		t.Helper()
		file, err := os.Open(filepath.Join(refDir, name+".f32.gz"))
		if err != nil {
			t.Fatal(err)
		}
		defer file.Close()
		gz, err := gzip.NewReader(file)
		if err != nil {
			t.Fatal(err)
		}
		defer gz.Close()
		data, err := io.ReadAll(gz)
		if err != nil || len(data) != rows*encoderWidth*4 {
			t.Fatalf("%s bytes=%d err=%v", name, len(data), err)
		}
		values := make([]float32, rows*encoderWidth)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
		return values
	}
	input := read("input")
	for _, lookahead := range []int{0, 3} {
		reference := read(fmt.Sprintf("look%d", lookahead))
		stream := &CachedEncoderTower{Tower: tower}
		for step := 0; step < rows/4; step++ {
			start := step * 4 * encoderWidth
			if step == 1 {
				before := stream.states[0].Conv.Snapshot()
				bad := append([]float32(nil), input[start:start+4*encoderWidth]...)
				bad[0] = float32(math.NaN())
				if _, err := stream.ForwardChunk(bad, 4, lookahead); err == nil || stream.seen != 4 || !reflect.DeepEqual(before, stream.states[0].Conv.Snapshot()) {
					t.Fatal("rejected chunk changed tower state")
				}
				missing := tower.layers[23]
				tower.layers[23] = nil
				if _, err := stream.ForwardChunk(input[start:start+4*encoderWidth], 4, lookahead); err == nil || stream.seen != 4 || !reflect.DeepEqual(before, stream.states[0].Conv.Snapshot()) {
					t.Fatal("late layer failure changed tower state")
				}
				tower.layers[23] = missing
			}
			got, err := stream.ForwardChunk(input[start:start+4*encoderWidth], 4, lookahead)
			if err != nil {
				t.Fatalf("look=%d step=%d: %v", lookahead, step, err)
			}
			var outside int
			var maxAbs, sumAbs float64
			for i, v := range got {
				want := reference[start+i]
				diff := math.Abs(float64(v - want))
				maxAbs = math.Max(maxAbs, diff)
				sumAbs += diff
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || diff > 3e-4+2e-5*math.Abs(float64(want)) {
					outside++
				}
			}
			mean := sumAbs / float64(len(got))
			t.Logf("look=%d step=%d max_abs=%g mean_abs=%g outside=%d", lookahead, step, maxAbs, mean, outside)
			if outside != 0 || mean > 2e-5 {
				t.Fatal("cached encoder tower differs from PyTorch")
			}
			if stream.seen != (step+1)*4 {
				t.Fatal("tower cache did not advance")
			}
		}
	}
}
