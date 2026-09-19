package community1

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type segmentationLoadOracle struct {
	Config  SegmentationLoadConfig
	Tensors map[string]struct {
		DType  string
		Shape  []int
		Values []float32
	}
	Frames           int
	Features         []float32
	LogProbabilities []float32 `json:"log_probabilities"`
}

func segmentationLoadOracles(t *testing.T) []segmentationLoadOracle {
	t.Helper()
	data, err := os.ReadFile("testdata/segmentation-load-reference.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	z, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	data, err = io.ReadAll(io.LimitReader(z, 16<<20))
	z.Close()
	if err != nil || len(data) >= 16<<20 {
		t.Fatal("segmentation fixture bound", err)
	}
	var f struct {
		Schema int
		Hashes map[string]string `json:"source_hashes"`
		Cases  []segmentationLoadOracle
	}
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	if f.Schema != 1 || len(f.Cases) != 3 || f.Hashes["pyannet"] != "3ceebc8c00e83d96747a706212102e7e99c44732ff1fc769e241e8de47d3d6af" || f.Hashes["sincnet"] != "3f151a2482c3f8c266b1efd9bdcab297238c52b4792a7265261f2da063e5dade" {
		t.Fatal("segmentation schema reference changed")
	}
	return f.Cases
}
func segmentationFixtureSource(c segmentationLoadOracle) *resnetSource {
	s := &resnetSource{infos: make(map[string]safetensors.TensorInfo), values: make(map[string][]float32)}
	names := make([]string, 0, len(c.Tensors))
	for name := range c.Tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	offset := 0
	for _, name := range names {
		v := c.Tensors[name]
		size := 4 * len(v.Values)
		s.infos[name] = safetensors.TensorInfo{DType: v.DType, Shape: append([]int(nil), v.Shape...), DataOffsets: [2]int{offset, offset + size}}
		offset += size
		s.values[name] = append([]float32(nil), v.Values...)
	}
	return s
}
func TestSegmentationLoadSourceSchemaAndFeatures(t *testing.T) {
	for index, c := range segmentationLoadOracles(t) {
		t.Run(fmt.Sprint(index), func(t *testing.T) {
			source := segmentationFixtureSource(c)
			// Every payload call reuses storage. The loader must bind/copy immediately.
			source.reuseValues = true
			model, err := LoadSegmentationSource(context.Background(), source, c.Config)
			if err != nil {
				t.Fatal(err)
			}
			if len(source.seen) != []int{38, 36, 22}[index] {
				t.Fatal("read inventory", len(source.seen))
			}
			for _, mode := range []LSTMMode{LSTMScalar, LSTMSIMD} {
				out, err := model.ForwardFeatures(context.Background(), c.Features, c.Frames, mode, HeadMode(mode))
				if err != nil {
					t.Fatal(err)
				}
				compareSincNet(t, "loaded feature/head", out, c.LogProbabilities, 2e-6)
			}
			// Independent tensor-to-struct wiring verifies every recurrent/head tensor,
			// including biases whose numerical contributions could be masked by a gate.
			recurrent := make([]LSTMLayer, c.Config.LSTM.NumLayers)
			for i := range recurrent {
				for dir := 0; dir < lstmDirections(c.Config.LSTM); dir++ {
					name := fmt.Sprintf("lstm.%%s_l%d", i)
					if c.Config.SplitLSTM {
						name = fmt.Sprintf("lstm.%d.%%s_l0", i)
					}
					if dir == 1 {
						name += "_reverse"
					}
					w := LSTMWeights{c.Tensors[fmt.Sprintf(name, "weight_ih")].Values, c.Tensors[fmt.Sprintf(name, "weight_hh")].Values, c.Tensors[fmt.Sprintf(name, "bias_ih")].Values, c.Tensors[fmt.Sprintf(name, "bias_hh")].Values}
					if dir == 0 {
						recurrent[i].Forward = w
					} else {
						recurrent[i].Reverse = w
					}
				}
			}
			expected, err := NewLSTM(context.Background(), c.Config.LSTM, recurrent)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(model.recurrent, expected) {
				t.Fatal("recurrent binding")
			}
			linears := make([]HeadLinear, c.Config.Head.NumLayers)
			for i := range linears {
				linears[i] = HeadLinear{c.Tensors[fmt.Sprintf("linear.%d.weight", i)].Values, c.Tensors[fmt.Sprintf("linear.%d.bias", i)].Values}
			}
			head, err := NewSegmentationHead(context.Background(), c.Config.Head, linears, HeadLinear{c.Tensors["classifier.weight"].Values, c.Tensors["classifier.bias"].Values})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(model.head, head) {
				t.Fatal("head binding")
			}
			getNorm := func(base string) SincNetNorm {
				return SincNetNorm{c.Tensors[base+".weight"].Values, c.Tensors[base+".bias"].Values}
			}
			sinc := SincNetWeights{WaveNorm: getNorm("sincnet.wav_norm1d"), LowHz: c.Tensors["sincnet.conv1d.0.filterbank.low_hz_"].Values, BandHz: c.Tensors["sincnet.conv1d.0.filterbank.band_hz_"].Values}
			for i := range sinc.Conv {
				sinc.Conv[i] = SincNetConv{c.Tensors[fmt.Sprintf("sincnet.conv1d.%d.weight", i+1)].Values, c.Tensors[fmt.Sprintf("sincnet.conv1d.%d.bias", i+1)].Values}
			}
			for i := range sinc.Norm {
				sinc.Norm[i] = getNorm(fmt.Sprintf("sincnet.norm1d.%d", i))
			}
			if !reflect.DeepEqual(model.sincnet, sinc) {
				t.Fatal("sincnet binding")
			}
			for _, v := range source.values {
				for i := range v {
					v[i] = 999
				}
			}
			for i := range source.reuse {
				source.reuse[i] = 999
			}
			if !reflect.DeepEqual(model.sincnet, sinc) {
				t.Fatal("frontend source alias")
			}
			out, err := model.ForwardFeatures(context.Background(), c.Features, c.Frames, LSTMSIMD, HeadSIMD)
			if err != nil {
				t.Fatal(err)
			}
			compareSincNet(t, "owned features", out, c.LogProbabilities, 2e-6)
			grid, err := model.Grid(1600)
			want, err2 := sincNetGrid(1600, c.Config.SincNetStride)
			if err != nil || err2 != nil || grid != want {
				t.Fatal("checkpoint grid")
			}
			// Structural guard: no assembled PCM or frontend accessor on public type.
			methods := reflect.TypeOf(model)
			for _, name := range []string{"Forward", "ForwardPCM", "Frontend", "SincNet"} {
				if _, ok := methods.MethodByName(name); ok {
					t.Fatal("frontend hold bypassed", name)
				}
			}
		})
	}
}
func TestSegmentationLoadMetadataBeforePayload(t *testing.T) {
	c := segmentationLoadOracles(t)[0]
	for _, kind := range []string{"missing", "extra", "shape", "dtype", "fixed_dtype", "extent", "negative", "overlap", "split", "width", "stride", "lstm", "head"} {
		source := segmentationFixtureSource(c)
		cfg := c.Config
		key := "classifier.weight"
		info := source.infos[key]
		switch kind {
		case "missing":
			delete(source.infos, key)
		case "extra":
			source.infos["training.weight"] = info
		case "shape":
			info.Shape = []int{info.Shape[1], info.Shape[0]}
			source.infos[key] = info
		case "dtype":
			info.DType = "I32"
			source.infos[key] = info
		case "fixed_dtype":
			key = "sincnet.conv1d.0.filterbank.n_"
			info = source.infos[key]
			info.DType = "F16"
			source.infos[key] = info
		case "extent":
			info.DataOffsets[1]--
			source.infos[key] = info
		case "negative":
			info.DataOffsets[0] = -1
			source.infos[key] = info
		case "overlap":
			size := info.DataOffsets[1] - info.DataOffsets[0]
			info.DataOffsets = [2]int{0, size}
			source.infos[key] = info
		case "split":
			cfg.SplitLSTM = true
		case "width":
			cfg.Head.InputSize++
		case "stride":
			cfg.SincNetStride = 11
		case "lstm":
			cfg.LSTM.HiddenSize = int(^uint(0) >> 1)
		case "head":
			cfg.Head.MaxActive = 9
		}
		model, err := LoadSegmentationSource(context.Background(), source, cfg)
		if err == nil || model != nil || len(source.seen) != 0 {
			t.Fatal("metadata read/accepted", kind, err, len(source.seen))
		}
		if kind == "dtype" && (!strings.Contains(err.Error(), "I32") || !strings.Contains(err.Error(), key)) {
			t.Fatal("lost dtype diagnostic", err)
		}
	}
	if model, err := LoadSegmentationSource(context.Background(), nil, c.Config); err == nil || model != nil {
		t.Fatal("nil source")
	}
}
func TestSegmentationLoadPayloadRejection(t *testing.T) {
	c := segmentationLoadOracles(t)[0]
	sentinel := errors.New("synthetic read error")
	for _, kind := range []string{"short", "shape_change", "nan", "infinite", "window", "time", "band", "read_error", "cancel"} {
		source := segmentationFixtureSource(c)
		key := "sincnet.wav_norm1d.weight"
		ctx, cancel := context.WithCancel(context.Background())
		switch kind {
		case "short":
			source.values[key] = nil
		case "shape_change":
			source.after = func(name string) { info := source.infos[name]; info.Shape = []int{1, 1}; source.infos[name] = info }
		case "nan":
			source.values[key][0] = float32(math.NaN())
		case "infinite":
			source.values[key][0] = float32(math.Inf(1))
		case "window":
			source.values["sincnet.conv1d.0.filterbank.window_"][2] = 0
		case "time":
			source.values["sincnet.conv1d.0.filterbank.n_"][2] = 0
		case "band":
			source.values["sincnet.conv1d.0.filterbank.low_hz_"][0] = 8000
		case "read_error":
			source.fail = sentinel
		case "cancel":
			source.after = func(string) { cancel() }
		}
		model, err := LoadSegmentationSource(ctx, source, c.Config)
		cancel()
		if err == nil || model != nil {
			t.Fatal("accepted payload", kind)
		}
		if kind == "read_error" && !errors.Is(err, sentinel) {
			t.Fatal("source cause")
		}
		if kind == "cancel" && !errors.Is(err, context.Canceled) {
			t.Fatal("cancel cause")
		}
	}
	for _, model := range []*SegmentationCheckpoint{nil, {}} {
		if out, err := model.ForwardFeatures(context.Background(), nil, 1, LSTMScalar, HeadScalar); err == nil || out != nil {
			t.Fatal("nil/zero features")
		}
		if _, err := model.Grid(1600); err == nil {
			t.Fatal("nil/zero grid")
		}
	}
}
func writeSegmentationSafe(t *testing.T, c segmentationLoadOracle, dtype string) (string, *resnetSource) {
	t.Helper()
	source := segmentationFixtureSource(c)
	names := make([]string, 0, len(source.infos))
	for name := range source.infos {
		names = append(names, name)
	}
	sort.Strings(names)
	var payload []byte
	for _, name := range names {
		info := source.infos[name]
		start := len(payload)
		info.DType = dtype
		if strings.HasSuffix(name, "window_") || strings.HasSuffix(name, "n_") {
			info.DType = "F32"
		}
		for i, value := range source.values[name] {
			switch info.DType {
			case "F32":
				payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(value))
			case "F16":
				bits := half.F32ToF16(value)
				payload = binary.LittleEndian.AppendUint16(payload, bits)
				source.values[name][i] = half.F16ToF32(bits)
			case "BF16":
				bits := uint16(math.Float32bits(value) >> 16)
				payload = binary.LittleEndian.AppendUint16(payload, bits)
				source.values[name][i] = half.BF16ToF32(bits)
			}
		}
		info.DataOffsets = [2]int{start, len(payload)}
		source.infos[name] = info
	}
	header, err := json.Marshal(source.infos)
	if err != nil {
		t.Fatal(err)
	}
	for len(header)%8 != 0 {
		header = append(header, ' ')
	}
	data := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	data = append(data, header...)
	data = append(data, payload...)
	path := filepath.Join(t.TempDir(), "synthetic.safetensors")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	return path, source
}
func TestSegmentationLoadSafetensorsDTypesAndClose(t *testing.T) {
	c := segmentationLoadOracles(t)[0]
	for _, dtype := range []string{"F32", "F16", "BF16"} {
		path, source := writeSegmentationSafe(t, c, dtype)
		file, err := safetensors.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		model, err := LoadSegmentationSource(context.Background(), file, c.Config)
		file.Close()
		if err != nil {
			t.Fatal(dtype, err)
		}
		expected, err := LoadSegmentationSource(context.Background(), source, c.Config)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(model, expected) {
			t.Fatal("safe dtype binding", dtype)
		}
		out, err := model.ForwardFeatures(context.Background(), c.Features, c.Frames, LSTMSIMD, HeadSIMD)
		if err != nil {
			t.Fatal(err)
		}
		reference, err := expected.ForwardFeatures(context.Background(), c.Features, c.Frames, LSTMSIMD, HeadSIMD)
		if err != nil {
			t.Fatal(err)
		}
		compareSincNet(t, "closed source output", out, reference, 0)
	}
}
func TestSegmentationLoadCancellationConcurrency(t *testing.T) {
	c := segmentationLoadOracles(t)[0]
	source := segmentationFixtureSource(c)
	count := newPowersetContext(0)
	_, err := LoadSegmentationSource(count, source, c.Config)
	count.cancel()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("segmentation load checkpoints%d", count.calls)
	for at := 1; at <= count.calls; at++ {
		ctx := newPowersetContext(at)
		out, err := LoadSegmentationSource(ctx, source, c.Config)
		ctx.cancel()
		if out != nil || !errors.Is(err, context.Canceled) {
			t.Fatal("load cancellation", at, err)
		}
	}
	model, err := LoadSegmentationSource(context.Background(), source, c.Config)
	if err != nil {
		t.Fatal(err)
	}
	count = newPowersetContext(0)
	_, err = model.ForwardFeatures(count, c.Features, c.Frames, LSTMSIMD, HeadSIMD)
	count.cancel()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("segmentation feature checkpoints%d", count.calls)
	for at := 1; at <= count.calls; at++ {
		ctx := newPowersetContext(at)
		out, err := model.ForwardFeatures(ctx, c.Features, c.Frames, LSTMSIMD, HeadSIMD)
		ctx.cancel()
		if out != nil || !errors.Is(err, context.Canceled) {
			t.Fatal("feature cancellation", at, err)
		}
	}
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out, err := model.ForwardFeatures(context.Background(), c.Features, c.Frames, LSTMSIMD, HeadSIMD)
			if err != nil {
				t.Error(err)
				return
			}
			compareSincNet(t, "concurrent loadedfeatures", out, c.LogProbabilities, 2e-6)
		}()
	}
	wg.Wait()
}
