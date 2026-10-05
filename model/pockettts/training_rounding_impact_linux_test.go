//go:build linux

package pockettts

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// The caller pins the entire checkpoint before streaming. Only inference
// parameters/buffers/EMA are materialised; unrelated Adam storage is skipped.
func roundingInferenceValues(path string) (map[string][]float32, map[string][]float32, error) {
	r, err := sharedOpenCheckpointP(path)
	if err != nil {
		return nil, nil, err
	}
	defer r.file.Close()
	params, ema := map[string][]float32{}, map[string][]float32{}
	var buffer [32 << 10]byte
	for group := 0; group < 4; group++ {
		if group > 0 {
			var raw [4]byte
			if _, err = io.ReadFull(r.file, raw[:]); err != nil {
				return nil, nil, err
			}
			r.count = int(binary.LittleEndian.Uint32(raw[:]))
			if r.count < 0 || r.count > maxBinaryTrainingTensors {
				return nil, nil, fmt.Errorf("inference group count")
			}
		}
		seen := map[string]bool{}
		for i := 0; i < r.count; i++ {
			name, n, err := r.next()
			if err != nil {
				return nil, nil, err
			}
			if seen[name] {
				return nil, nil, fmt.Errorf("duplicate inference tensor")
			}
			seen[name] = true
			if group == 2 {
				if _, err = r.file.Seek(int64(n)*8, io.SeekCurrent); err != nil {
					return nil, nil, err
				}
				continue
			}
			values := make([]float32, n)
			for start := 0; start < n; {
				count := min(n-start, len(buffer)/4)
				if _, err = io.ReadFull(r.file, buffer[:count*4]); err != nil {
					return nil, nil, err
				}
				for j := 0; j < count; j++ {
					values[start+j] = math.Float32frombits(binary.LittleEndian.Uint32(buffer[4*j:]))
				}
				start += count
			}
			if !finiteF32(values) {
				return nil, nil, fmt.Errorf("nonfinite inference values")
			}
			if group == 3 {
				ema[name] = values
			} else {
				params[name] = values
				if group == 1 {
					ema[name] = values
				}
			}
		}
	}
	return params, ema, nil
}

func TestRoundingInferenceCheckpointStream(t *testing.T) {
	r, _, _ := residentGraphRunner(t, "vulkan-resident-devicehandoff-hybrid-v1")
	state, err := r.trainer.State()
	if err != nil {
		t.Fatal(err)
	}
	path := t.TempDir() + "/state.bin"
	if err = SaveFullTrainingStateBinary(path, state); err != nil {
		t.Fatal(err)
	}
	p, e, err := roundingInferenceValues(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range state.Params {
		assertSliceClose(t, "stream P "+v.Name, p[v.Name], v.Values, 0)
	}
	for _, v := range state.Buffers {
		assertSliceClose(t, "stream B "+v.Name, p[v.Name], v.Values, 0)
		assertSliceClose(t, "stream EMA buffer "+v.Name, e[v.Name], v.Values, 0)
	}
	for _, v := range state.EMA {
		assertSliceClose(t, "stream EMA "+v.Name, e[v.Name], v.Values, 0)
	}
	// Short reads in values/groups cannot be accepted as an inference snapshot.
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []int{0, 15, 52, 100, len(raw) / 2, len(raw) - 40} {
		bad := t.TempDir() + "/short.bin"
		if err = os.WriteFile(bad, raw[:n], 0600); err != nil {
			t.Fatal(err)
		}
		if _, _, err = roundingInferenceValues(bad); err == nil {
			t.Fatal("truncated snapshot accepted", n)
		}
	}
}

// Selected P/EMA reads are diagnostic only and require the pinned whole file.
func roundingCheckpointPoints(path string, points map[string][]int, visit func(int, string, int, float32)) error {
	r, err := sharedOpenCheckpointP(path)
	if err != nil {
		return err
	}
	defer r.file.Close()
	info, err := r.file.Stat()
	if err != nil {
		return err
	}
	for group := 0; group < 4; group++ {
		if group > 0 {
			var raw [4]byte
			if _, err = io.ReadFull(r.file, raw[:]); err != nil {
				return err
			}
			r.count = int(binary.LittleEndian.Uint32(raw[:]))
			if r.count < 0 || r.count > maxBinaryTrainingTensors {
				return fmt.Errorf("point group count")
			}
		}
		seen := map[string]bool{}
		for i := 0; i < r.count; i++ {
			name, n, err := r.next()
			if err != nil {
				return err
			}
			if seen[name] {
				return fmt.Errorf("duplicate point tensor")
			}
			seen[name] = true
			start, err := r.file.Seek(0, io.SeekCurrent)
			if err != nil {
				return err
			}
			bytes := int64(n) * 4
			if group == 2 {
				bytes *= 2
			}
			if bytes > info.Size()-start {
				return io.ErrUnexpectedEOF
			}
			if group == 0 || group == 3 {
				for _, index := range points[name] {
					if index < 0 || index >= n {
						return fmt.Errorf("point index")
					}
					var raw [4]byte
					if _, err = r.file.ReadAt(raw[:], start+int64(index)*4); err != nil {
						return err
					}
					value := math.Float32frombits(binary.LittleEndian.Uint32(raw[:]))
					if !isFinite(value) {
						return fmt.Errorf("nonfinite point value")
					}
					visit(group, name, index, value)
				}
			}
			if _, err = r.file.Seek(bytes, io.SeekCurrent); err != nil {
				return err
			}
		}
		if group == 0 || group == 3 {
			for name := range points {
				if !seen[name] {
					return fmt.Errorf("point tensor missing")
				}
			}
		}
	}
	return nil
}

func TestRoundingCheckpointSelectedPoints(t *testing.T) {
	r, _, _ := residentGraphRunner(t, "vulkan-resident-devicehandoff-hybrid-v1")
	state, err := r.trainer.State()
	if err != nil {
		t.Fatal(err)
	}
	path := t.TempDir() + "/point.bin"
	if err = SaveFullTrainingStateBinary(path, state); err != nil {
		t.Fatal(err)
	}
	name := state.Params[0].Name
	n := len(state.Params[0].Values)
	seen := 0
	if err = roundingCheckpointPoints(path, map[string][]int{name: {0, n - 1}}, func(group int, got string, index int, value float32) {
		seen++
		want := r.trainer.params[got][index]
		if group == 3 {
			want = r.trainer.EMA[got][index]
		}
		if value != want {
			t.Fatal("point association")
		}
	}); err != nil || seen != 4 {
		t.Fatal("points", err, seen)
	}
	for _, selection := range []map[string][]int{{name: {-1}}, {name: {n}}, {"missing": {0}}} {
		if err = roundingCheckpointPoints(path, selection, func(int, string, int, float32) {}); err == nil {
			t.Fatal("invalid selection")
		}
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	short := t.TempDir() + "/short.bin"
	if err = os.WriteFile(short, raw[:len(raw)-40], 0600); err != nil {
		t.Fatal(err)
	}
	if err = roundingCheckpointPoints(short, map[string][]int{name: {0}}, func(int, string, int, float32) {}); err == nil {
		t.Fatal("short point snapshot")
	}
}

// Freeze step-two CPU clipped gradients to measure optimizer memory alone.
// No changed gradient is fed into a real trainer; this is not native resume.
func TestPocketJointRoundingFixedGradientReplay(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_ROUNDING_IMPACT") != "1" {
		t.Skip("explicit fixed-gradient optimizer sensitivity")
	}
	paths := []string{
		releasedAsset(t, "GO_PHERENCE_POCKET_SHARED_FULL_CPU", "28d7d57d949fe8d9a2b2cb22c3f9d51773e775d442fc383fad7192b26f1cd960", 1431217112),
		releasedAsset(t, "GO_PHERENCE_POCKET_ROUNDING_CPU_STEP2", "6ade532407f069548a38efeabfb217267c721940fa801517326ecb51dee84dcc", 1431217112),
	}
	points := []struct {
		Name    string
		Index   int
		P, M, V float32
	}{
		{"flow_lm.transformer.layers.1.linear2.weight", 1914574, .003770981682, -1.07969857e-11, 5.828743534e-22},
		{"flow_lm.transformer.layers.2.linear2.weight", 120489, -.006497516297, 1.62243316e-10, 1.316144392e-19},
	}
	for _, point := range points {
		var p, e, m, v [2]float32
		for step, path := range paths {
			selected := map[string][]int{point.Name: {point.Index}}
			if err := roundingCheckpointPoints(path, selected, func(group int, name string, index int, value float32) {
				if group == 0 {
					p[step] = value
				} else {
					e[step] = value
				}
			}); err != nil {
				t.Fatal(err)
			}
			if err := integratedCPUMomentPoints(path, selected, func(name string, index int, mm, vv float32) { m[step], v[step] = mm, vv }); err != nil {
				t.Fatal(err)
			}
		}
		config := DefaultAdamWConfig()
		g := float32((float64(m[1]) - float64(config.Beta1)*float64(m[0])) / float64(1-config.Beta1))
		update := func(parameter, mm, vv float32) (float32, float32, float32) {
			mm = config.Beta1*mm + (1-config.Beta1)*g
			vv = config.Beta2*vv + (1-config.Beta2)*g*g
			mHat := mm / float32(1-math.Pow(float64(config.Beta1), 2))
			vHat := vv / float32(1-math.Pow(float64(config.Beta2), 2))
			parameter *= 1 - config.LearningRate*config.WeightDecay
			parameter -= config.LearningRate * mHat / (float32(math.Sqrt(float64(vHat))) + config.Epsilon)
			return parameter, mm, vv
		}
		cpuP, cpuM, cpuV := update(p[0], m[0], v[0])
		nativeP, nativeM, nativeV := update(point.P, point.M, point.V)
		// Inverting rounded M to G cannot promise exact recovery; bound the
		// reconstruction at two F32 ULPs, not by the optimizer parity tolerance.
		bound := 2 * math.Abs(float64(math.Nextafter32(p[1], float32(math.Inf(1))))-float64(p[1]))
		if math.Abs(float64(cpuP)-float64(p[1])) > bound {
			t.Fatal("CPU moment inversion does not reproduce reference", cpuP, p[1], bound)
		}
		initialEMA := e[0] + (1-float32(.999))*(point.P-p[0])
		nativeEMA := float32(.999)*initialEMA + (1-float32(.999))*nativeP
		t.Logf("fixed-gradient step2 %s[%d] clippedG=%g CPU_P_recovery=%g step1_P_delta=%g step2_P_delta=%g step2_EMA_delta=%g M_delta=%g V_delta=%g above4e6=%t", point.Name, point.Index, g, float64(cpuP)-float64(p[1]), float64(point.P)-float64(p[0]), float64(nativeP)-float64(p[1]), float64(nativeEMA)-float64(e[1]), float64(nativeM)-float64(cpuM), float64(nativeV)-float64(cpuV), math.Abs(float64(nativeP)-float64(p[1])) > 4e-6)
	}
}

// This is a controlled two-coordinate perturbation experiment, NOT a native
// model comparison. All other CPU-checkpoint values stay fixed; no export,
// checkpoint, optimizer publication or numerical gate exemption is allowed.
func TestPocketJointRoundingInferenceImpact(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_ROUNDING_IMPACT") != "1" {
		t.Skip("explicit independent checkpoint sensitivity diagnostic")
	}
	cpuPath := releasedAsset(t, "GO_PHERENCE_POCKET_SHARED_FULL_CPU", "28d7d57d949fe8d9a2b2cb22c3f9d51773e775d442fc383fad7192b26f1cd960", 1431217112)
	frozenPath := releasedAsset(t, "GO_PHERENCE_POCKETTTS_VOICE_MODEL", "fb0dc01b0d4d2e1c905b7a3e0676e3d9c96d5ae460e24e3ab94981805babf997", 219029196)
	params, ema, err := roundingInferenceValues(cpuPath)
	if err != nil {
		t.Fatal(err)
	}
	frozen, err := safetensors.Open(frozenPath)
	if err != nil {
		t.Fatal(err)
	}
	defer frozen.Close()
	cfg := releasedConfig(t)
	mimi, err := LoadMimiDecoderCPU(frozen, cfg)
	if err != nil {
		t.Fatal(err)
	}
	points := []struct {
		Name                       string
		Index                      int
		CPU, Native, CPUM, NativeM float32
	}{
		{"flow_lm.transformer.layers.1.linear2.weight", 1914574, .00376618444, .003770981682, 1.348497182e-11, -1.07969857e-11},
		{"flow_lm.transformer.layers.2.linear2.weight", 120489, -.006493393797, -.006497516297, 1.350521084e-10, 1.62243316e-10},
	}
	nativeEMA := make([]float32, len(points))
	for i, p := range points {
		if params[p.Name][p.Index] != p.CPU {
			t.Fatal("pinned outlier association", p.Name, params[p.Name][p.Index], p.CPU)
		}
		initial, _, err := frozen.GetFloat32(p.Name)
		if err != nil {
			t.Fatal(err)
		}
		nativeEMA[i] = float32(.999)*initial[p.Index] + (1-float32(.999))*p.Native
		t.Logf("point=%s[%d] P_delta=%g EMA_delta=%g clippedG_CPU=%g clippedG_native=%g", p.Name, p.Index, float64(p.Native)-float64(p.CPU), float64(nativeEMA[i])-float64(ema[p.Name][p.Index]), p.CPUM/(1-float32(.9)), p.NativeM/(1-float32(.9)))
		if math.Abs(float64(nativeEMA[i])-float64(ema[p.Name][p.Index])) > 6e-8 {
			t.Fatal("two-point EMA storage gate exceeded")
		}
	}
	for _, mode := range []string{"parameters", "EMA"} {
		values := params
		if mode == "EMA" {
			values = ema
		}
		lm, flow := savedEMAInferenceModels(t, cfg, values)
		generator, err := NewGeneratorCPU(lm, flow, mimi, values["flow_lm.emb_mean"], values["flow_lm.emb_std"])
		if err != nil {
			t.Fatal(err)
		}
		cases := [][]uint32{{1, 7, 23}, {23, 7, 1}, make([]uint32, 512)}
		for i := range cases[2] {
			cases[2][i] = uint32(1 + i%512)
		}
		// Each observation is owned and bounded. Reset session state on every request.
		type observed struct {
			hidden, velocity, pcm []float32
			eos                   float32
		}
		observe := func(tokens []uint32) observed {
			hidden, eos, err := lm.Prefill(tokens, lm.BOS)
			if err != nil {
				t.Fatal(err)
			}
			velocity := make([]float32, cfg.Mimi.InnerDim)
			input := make([]float32, len(velocity))
			for i := range input {
				input[i] = float32(math.Sin(float64(i+1)*.37) * .3)
			}
			if err = flow.Forward(velocity, hidden, []float32{0, .5}, input); err != nil {
				t.Fatal(err)
			}
			template, err := lm.Transformer.NewState(len(tokens) + 4)
			if err != nil {
				t.Fatal(err)
			}
			session, err := NewSession(generator, template, 4)
			if err != nil {
				t.Fatal(err)
			}
			noise := func(frame int, dst []float32) error {
				for i := range dst {
					dst[i] = float32(math.Sin(float64(i+1)*.37+float64(frame)*.11) * .3)
				}
				return nil
			}
			pcm := make([]float32, 4*SamplesPerFrame)
			n, err := session.GenerateInto(pcm, tokens, 4, 5, 2, float32(math.Inf(1)), noise)
			if err != nil || n != len(pcm) {
				t.Fatal("sensitivity generation", n, err)
			}
			// A second request must preserve reset semantics, with unchanged voice template.
			again := make([]float32, len(pcm))
			n, err = session.GenerateInto(again, tokens, 4, 5, 2, float32(math.Inf(1)), noise)
			if err != nil || n != len(pcm) {
				t.Fatal("reset sensitivity", err)
			}
			savedEMAErrorDistribution(t, mode+" reset", again, pcm, 3e-5)
			if template.Position != 0 {
				t.Fatal("mutated template")
			}
			return observed{append([]float32(nil), hidden...), velocity, pcm, eos}
		}
		before := make([]observed, len(cases))
		for i, tokens := range cases {
			before[i] = observe(tokens)
		}
		saved := make([]float32, len(points))
		for i, p := range points {
			saved[i] = values[p.Name][p.Index]
			if mode == "EMA" {
				values[p.Name][p.Index] = nativeEMA[i]
			} else {
				values[p.Name][p.Index] = p.Native
			}
		}
		for i, tokens := range cases {
			after := observe(tokens)
			label := fmt.Sprintf("%s case%d tokens%d", mode, i, len(tokens))
			savedEMAErrorDistribution(t, label+" hidden", after.hidden, before[i].hidden, 3e-5)
			savedEMAErrorDistribution(t, label+" EOS", []float32{after.eos}, []float32{before[i].eos}, 3e-5)
			savedEMAErrorDistribution(t, label+" velocity", after.velocity, before[i].velocity, 2e-5)
			savedEMAErrorDistribution(t, label+" PCM", after.pcm, before[i].pcm, 3e-5)
		}
		for i, p := range points {
			values[p.Name][p.Index] = saved[i]
		}
	}
}
