//go:build linux

package pockettts

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

type sharedCheckpointPReader struct {
	file   *os.File
	header [36]byte
	count  int
}

func sharedOpenCheckpointP(path string) (*sharedCheckpointPReader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	fail := func(e error) (*sharedCheckpointPReader, error) { f.Close(); return nil, e }
	var magic [16]byte
	if _, err = io.ReadFull(f, magic[:]); err != nil {
		return fail(err)
	}
	if magic != fullTrainingBinaryMagic {
		return fail(fmt.Errorf("P source magic"))
	}
	r := &sharedCheckpointPReader{file: f}
	if _, err = io.ReadFull(f, r.header[:]); err != nil {
		return fail(err)
	}
	var b [4]byte
	if _, err = io.ReadFull(f, b[:]); err != nil {
		return fail(err)
	}
	r.count = int(binary.LittleEndian.Uint32(b[:]))
	if r.count <= 0 || r.count > maxBinaryTrainingTensors {
		return fail(fmt.Errorf("P source count"))
	}
	return r, nil
}
func (r *sharedCheckpointPReader) next() (string, int, error) {
	var b [8]byte
	if _, err := io.ReadFull(r.file, b[:2]); err != nil {
		return "", 0, err
	}
	n := int(binary.LittleEndian.Uint16(b[:2]))
	if n <= 0 || n > maxBinaryTrainingNameBytes {
		return "", 0, fmt.Errorf("P source name length")
	}
	name := make([]byte, n)
	if _, err := io.ReadFull(r.file, name); err != nil {
		return "", 0, err
	}
	if _, err := io.ReadFull(r.file, b[:]); err != nil {
		return "", 0, err
	}
	length := binary.LittleEndian.Uint64(b[:])
	if length == 0 || length > 1<<26 {
		return "", 0, fmt.Errorf("P source extent")
	}
	return string(name), int(length), nil
}

type sharedPDistribution struct {
	Values, AboveGate, OriginalOutliersAboveGate, NewOutliersAboveGate int64
	MaxAbs, MeanAbs, RMS                                               float64
}

func sharedFullParameterDistribution(t *testing.T, tr *FullTrainer, g TrainingStepGradients, points []sharedSensitivityPoint, norm float64, root string) {
	t.Helper()
	cpuPath := releasedAsset(t, "GO_PHERENCE_POCKET_SHARED_FULL_CPU", "d0d366f789a6ab0078bd0f1534d16a38dddb4e08a3193797be23c9f6b1794e5c", 1431217112)
	nativePath := releasedAsset(t, "GO_PHERENCE_POCKET_SHARED_FULL_NATIVE", "bc4092eb94c8434f11332c04d4b95715862d913ef89087ac57a4b794852868fd", 1431217112)
	cpu, err := sharedOpenCheckpointP(cpuPath)
	if err != nil {
		t.Fatal(err)
	}
	defer cpu.file.Close()
	native, err := sharedOpenCheckpointP(nativePath)
	if err != nil {
		t.Fatal(err)
	}
	defer native.file.Close()
	if cpu.header != native.header || cpu.count != native.count || cpu.count != len(tr.bindings) || binary.LittleEndian.Uint64(cpu.header[4:12]) != 1 {
		t.Fatal("P source association/header")
	}
	c := tr.Config
	expected := []float32{c.LearningRate, c.Beta1, c.Beta2, c.Epsilon, c.WeightDecay, tr.EMADecay}
	for i, v := range expected {
		if math.Float32frombits(binary.LittleEndian.Uint32(cpu.header[12+i*4:])) != v {
			t.Fatal("P source optimizer config")
		}
	}
	original := map[string]map[int]bool{}
	for _, p := range points {
		if original[p.Name] == nil {
			original[p.Name] = map[int]bool{}
		}
		original[p.Name][p.Index] = true
	}
	grads, err := fullGradientMap(g)
	if err != nil {
		t.Fatal(err)
	}
	scale := float32(1)
	if norm > 1 {
		scale = float32(1 / norm)
	}
	report := struct {
		Scope                            string
		GradientNorm                     float64
		Native, CPU, OriginalNativeVsCPU sharedPDistribution
	}{Scope: "All parameters; host sensitivity prediction vs saved native/CPU, no optimizer or native execution. Input hashes are provenance only.", GradientNorm: norm}
	arrays := []*sharedPDistribution{&report.Native, &report.CPU, &report.OriginalNativeVsCPU}
	sum, square := [3]float64{}, [3]float64{}
	var a, b [32 << 10]byte
	for tensor := 0; tensor < cpu.count; tensor++ {
		name, n, err := cpu.next()
		if err != nil {
			t.Fatal(err)
		}
		other, m, err := native.next()
		if err != nil || other != name || m != n {
			t.Fatal("P source inventory", name, other, n, m, err)
		}
		initial, ok := tr.params[name]
		gradient, found := grads[name]
		if !ok || !found || len(initial) != n || len(gradient) != n {
			t.Fatal("complete predicted P inventory", name)
		}
		for offset := 0; offset < n; {
			count := min(n-offset, len(a)/4)
			if _, err := io.ReadFull(cpu.file, a[:4*count]); err != nil {
				t.Fatal(err)
			}
			if _, err := io.ReadFull(native.file, b[:4*count]); err != nil {
				t.Fatal(err)
			}
			for i := 0; i < count; i++ {
				at := offset + i
				cv := math.Float32frombits(binary.LittleEndian.Uint32(a[i*4:]))
				nv := math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
				x := gradient[at] * scale
				// Diagnostic first-step AdamW prediction; never assigned to model state.
				moment, variance := (1-c.Beta1)*x, ((1-c.Beta2)*x)*x
				predicted := initial[at] * (1 - c.LearningRate*c.WeightDecay)
				predicted -= c.LearningRate * (moment / (1 - c.Beta1)) / (float32(math.Sqrt(float64(variance/(1-c.Beta2)))) + c.Epsilon)
				if !isFinite(cv) || !isFinite(nv) || !isFinite(predicted) {
					t.Fatal("non-finite fullscan", name, at)
				}
				for j, d := range []float64{math.Abs(float64(predicted) - float64(nv)), math.Abs(float64(predicted) - float64(cv)), math.Abs(float64(cv) - float64(nv))} {
					r := arrays[j]
					r.Values++
					r.MaxAbs = max(r.MaxAbs, d)
					sum[j] += d
					square[j] += d * d
					if d > 4e-6 {
						r.AboveGate++
						if original[name][at] {
							r.OriginalOutliersAboveGate++
						} else {
							r.NewOutliersAboveGate++
						}
					}
				}
			}
			offset += count
		}
	}
	for j, r := range arrays {
		r.MeanAbs = sum[j] / float64(r.Values)
		r.RMS = math.Sqrt(square[j] / float64(r.Values))
	}
	if report.Native.Values != 89449730 || report.OriginalNativeVsCPU.AboveGate != 1293 || report.OriginalNativeVsCPU.NewOutliersAboveGate != 0 {
		t.Fatal("original full gate reproduction", report)
	}
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.OpenFile(filepath.Join(root, "full-parameter-distribution.json"), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		t.Fatal(err)
	}
	_, werr := file.Write(append(data, '\n'))
	cerr := file.Close()
	if werr != nil || cerr != nil {
		t.Fatal(werr, cerr)
	}
	t.Logf("full Pscan values=%d nativeAbove=%d originalRemaining=%d newAbove=%d max=%g rms=%g; CPUAbove=%d", report.Native.Values, report.Native.AboveGate, report.Native.OriginalOutliersAboveGate, report.Native.NewOutliersAboveGate, report.Native.MaxAbs, report.Native.RMS, report.CPU.AboveGate)
}

func TestPocketSharedCheckpointPReaderRejectsMalformed(t *testing.T) {
	for _, kind := range []string{"short-magic", "short-header", "short-count", "zero-count", "long-count", "short-name-length", "zero-name", "long-name", "short-name", "short-extent", "zero-extent", "long-extent"} {
		t.Run(kind, func(t *testing.T) {
			data := append([]byte(nil), fullTrainingBinaryMagic[:]...)
			data = append(data, make([]byte, 36+4)...)
			binary.LittleEndian.PutUint32(data[52:], 1)
			var tail [8]byte
			switch kind {
			case "short-magic":
				data = data[:3]
			case "short-header":
				data = data[:20]
			case "short-count":
				data = data[:54]
			case "zero-count":
				binary.LittleEndian.PutUint32(data[52:], 0)
			case "long-count":
				binary.LittleEndian.PutUint32(data[52:], maxBinaryTrainingTensors+1)
			case "short-name-length":
				data = append(data, 1)
			case "zero-name":
				data = append(data, 0, 0)
			case "long-name":
				binary.LittleEndian.PutUint16(tail[:], maxBinaryTrainingNameBytes+1)
				data = append(data, tail[:2]...)
			case "short-name":
				data = append(data, 2, 0, 'a')
			default:
				data = append(data, 1, 0, 'a')
				if kind == "short-extent" {
					data = append(data, 1)
				} else {
					length := uint64(0)
					if kind == "long-extent" {
						length = (1 << 26) + 1
					}
					binary.LittleEndian.PutUint64(tail[:], length)
					data = append(data, tail[:]...)
				}
			}
			path := filepath.Join(t.TempDir(), "malformed.bin")
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
			r, err := sharedOpenCheckpointP(path)
			if r == nil {
				if err == nil {
					t.Fatal("nil reader without error")
				}
				return
			}
			defer r.file.Close()
			if _, _, err := r.next(); err == nil {
				t.Fatal("malformed tensor admitted")
			}
		})
	}
}

func TestPocketSharedCheckpointPReader(t *testing.T) {
	state := binaryTrainingFixture(t)
	path := filepath.Join(t.TempDir(), "state.bin")
	if err := SaveFullTrainingStateBinary(path, state); err != nil {
		t.Fatal(err)
	}
	r, err := sharedOpenCheckpointP(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.file.Close()
	if r.count != len(state.Params) {
		t.Fatal("count")
	}
	for _, p := range state.Params {
		name, n, err := r.next()
		if err != nil || name != p.Name || n != len(p.Values) {
			t.Fatal("inventory", err)
		}
		raw := make([]byte, 4*n)
		if _, err := io.ReadFull(r.file, raw); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, n)
		for i := range got {
			got[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		if !reflect.DeepEqual(got, p.Values) {
			t.Fatal("persistence reader")
		}
	}
	if _, err := sharedOpenCheckpointP(path + "missing"); err == nil {
		t.Fatal("missing source")
	}
	bad := filepath.Join(t.TempDir(), "bad")
	if err := os.WriteFile(bad, make([]byte, 56), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := sharedOpenCheckpointP(bad); err == nil {
		t.Fatal("bad magic")
	}
}
