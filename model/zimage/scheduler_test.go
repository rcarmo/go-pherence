package zimage

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

func TestFlowMatchPinnedReference(t *testing.T) {
	data, err := os.ReadFile("testdata/flowmatch_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle struct {
		Schema                int                                `json:"schema"`
		ModelRevision         string                             `json:"model_revision"`
		SchedulerConfigSHA256 string                             `json:"scheduler_config_sha256"`
		ReferenceSource       string                             `json:"reference_source"`
		SchedulerSourceSHA256 string                             `json:"scheduler_source_sha256"`
		PipelineSourceSHA256  string                             `json:"pipeline_source_sha256"`
		NumPy1000SigmaSHA256  string                             `json:"numpy_1000_sigma_sha256"`
		NumPy1000TimeSHA256   string                             `json:"numpy_1000_timestep_sha256"`
		Config                loaderconfig.ZImageSchedulerConfig `json:"config"`
		Steps                 int                                `json:"steps"`
		Sigmas                []float32                          `json:"sigmas"`
		Timesteps             []float32                          `json:"timesteps"`
		Sample                []float32                          `json:"sample"`
		Velocity              []float32                          `json:"velocity"`
		FirstStep             []float32                          `json:"first_step"`
	}
	if err := json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Schema != 1 || oracle.ModelRevision != "f332072aa78be7aecdf3ee76d5c247082da564a6" || oracle.SchedulerConfigSHA256 != "3b979ab0956e4f5e8d02ec409ac6a4ece1555191d15bd10788fdc85fea5d13fc" || oracle.ReferenceSource != "huggingface/diffusers@0377f0c1b34e3ff313d41edad1bd79c2ed8bb5ec" || oracle.SchedulerSourceSHA256 != "1af27be5b2f92b7d139d3c50239be5ce3eafc4a7eddf59c2d30689f8ec31e93d" || oracle.PipelineSourceSHA256 != "2e31dfe83498a963895952053dbe939840f67a30c747b11ed5ab6f71751d6a92" {
		t.Fatal("unexpected scheduler reference provenance")
	}
	got, err := DefaultFlowSchedule(oracle.Config, oracle.Steps)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Sigmas, oracle.Sigmas) || !reflect.DeepEqual(got.Timesteps, oracle.Timesteps) {
		t.Fatalf("schedule=%+v want sigmas=%v timesteps=%v", got, oracle.Sigmas, oracle.Timesteps)
	}
	out := make([]float32, len(oracle.Sample))
	if err := EulerStepInto(out, oracle.Sample, oracle.Velocity, got.Sigmas[0], got.Sigmas[1]); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(out, oracle.FirstStep) {
		t.Fatalf("Euler=%v want=%v", out, oracle.FirstStep)
	}
	full, err := DefaultFlowSchedule(oracle.Config, 1000)
	if err != nil {
		t.Fatal(err)
	}
	if got := float32SHA256(full.Sigmas); got != oracle.NumPy1000SigmaSHA256 {
		t.Fatalf("1000-step sigma hash=%s want=%s", got, oracle.NumPy1000SigmaSHA256)
	}
	if got := float32SHA256(full.Timesteps); got != oracle.NumPy1000TimeSHA256 {
		t.Fatalf("1000-step timestep hash=%s want=%s", got, oracle.NumPy1000TimeSHA256)
	}
	for _, n := range []int{1, 2, 8} {
		plan, err := DefaultFlowSchedule(oracle.Config, n)
		if err != nil || len(plan.Timesteps) != n || len(plan.Sigmas) != n+1 || plan.Sigmas[0] != 1 || plan.Sigmas[n] != 0 {
			t.Fatalf("steps=%d plan=%+v err=%v", n, plan, err)
		}
		for i := 0; i < n; i++ {
			if plan.Sigmas[i] <= plan.Sigmas[i+1] {
				t.Fatalf("non-descending sigmas: %v", plan.Sigmas)
			}
		}
	}
}

// Hash the independent NumPy F32 reference in a fixed little-endian format.
func float32SHA256(v []float32) string {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(x))
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func TestFlowMatchRejectsUnsupported(t *testing.T) {
	cfg := loaderconfig.ZImageSchedulerConfig{ClassName: "FlowMatchEulerDiscreteScheduler", NumTrainTimesteps: 1000, Shift: 3}
	for _, steps := range []int{0, -1, MaxFlowSteps + 1} {
		if _, err := DefaultFlowSchedule(cfg, steps); err == nil {
			t.Fatalf("accepted steps=%d", steps)
		}
	}
	for _, mutate := range []func(*loaderconfig.ZImageSchedulerConfig){
		func(c *loaderconfig.ZImageSchedulerConfig) { c.ClassName = "other" },
		func(c *loaderconfig.ZImageSchedulerConfig) { c.Shift = 0 },
		func(c *loaderconfig.ZImageSchedulerConfig) { c.Shift = math.NaN() },
		func(c *loaderconfig.ZImageSchedulerConfig) { c.Shift = math.MaxFloat64 },
		func(c *loaderconfig.ZImageSchedulerConfig) { c.UseDynamicShifting = true },
		func(c *loaderconfig.ZImageSchedulerConfig) { c.NumTrainTimesteps = 0 },
	} {
		bad := cfg
		mutate(&bad)
		if _, err := DefaultFlowSchedule(bad, 4); err == nil {
			t.Fatalf("accepted config=%+v", bad)
		}
	}
}

func TestFlowBuffersDisjoint(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	if overlap(nil, a) || overlap(a[:2], a[2:]) || !overlap(a[:2], a[1:3]) {
		t.Fatal("incorrect empty, adjacent, or overlapping buffer range")
	}
}

func TestEulerStepTransactionalAndAliasing(t *testing.T) {
	const sigma, sigmaNext = float32(1), float32(.5)
	for _, tc := range []struct{ sample, velocity, want []float32 }{
		{[]float32{1, 2}, []float32{2, 4}, []float32{0, 0}},
		{[]float32{2, -1}, []float32{-2, 2}, []float32{3, -2}},
	} {
		out := make([]float32, 2)
		if err := EulerStepInto(out, tc.sample, tc.velocity, sigma, sigmaNext); err != nil || !reflect.DeepEqual(out, tc.want) {
			t.Fatalf("Euler=%v err=%v", out, err)
		}
		inplace := append([]float32(nil), tc.sample...)
		if err := EulerStepInto(inplace, inplace, tc.velocity, sigma, sigmaNext); err != nil || !reflect.DeepEqual(inplace, tc.want) {
			t.Fatalf("in-place sample=%v err=%v", inplace, err)
		}
		inplace = append([]float32(nil), tc.velocity...)
		if err := EulerStepInto(inplace, tc.sample, inplace, sigma, sigmaNext); err != nil || !reflect.DeepEqual(inplace, tc.want) {
			t.Fatalf("in-place velocity=%v err=%v", inplace, err)
		}
	}
	dst := []float32{99, 99}
	sample := []float32{1, 2}
	velocity := []float32{2, float32(math.Inf(1))}
	if err := EulerStepInto(dst, sample, velocity, sigma, sigmaNext); err == nil || !reflect.DeepEqual(dst, []float32{99, 99}) {
		t.Fatalf("mutated rejected result: dst=%v err=%v", dst, err)
	}
	storage := []float32{1, 2, 3, 4}
	if err := EulerStepInto(storage[1:3], storage[:2], []float32{2, 2}, sigma, sigmaNext); err == nil || !reflect.DeepEqual(storage, []float32{1, 2, 3, 4}) {
		t.Fatalf("accepted partial overlap: %v err=%v", storage, err)
	}
	// Even an otherwise valid first element must not be written when a later
	// product overflows; validation is transactional for the entire vector.
	if err := EulerStepInto(dst, []float32{1, math.MaxFloat32}, []float32{1, -math.MaxFloat32}, sigma, sigmaNext); err == nil || !reflect.DeepEqual(dst, []float32{99, 99}) {
		t.Fatalf("accepted overflowing result or mutated dst: %v err=%v", dst, err)
	}
	for _, sigmas := range [][2]float32{{.5, .6}, {-1, 0}, {1, float32(math.NaN())}} {
		if err := EulerStepInto(dst, sample, []float32{2, 2}, sigmas[0], sigmas[1]); err == nil {
			t.Fatalf("accepted sigma=%v", sigmas)
		}
	}
	if err := EulerStepInto(dst, sample[:1], []float32{2, 2}, sigma, sigmaNext); err == nil {
		t.Fatal("accepted short sample")
	}
	if err := EulerStepInto(nil, nil, nil, sigma, sigmaNext); err == nil {
		t.Fatal("accepted empty buffers")
	}
	if err := EulerStepInto(dst, []float32{1, float32(math.NaN())}, []float32{2, 2}, sigma, sigmaNext); err == nil || !reflect.DeepEqual(dst, []float32{99, 99}) {
		t.Fatalf("accepted NaN sample or mutated dst: %v err=%v", dst, err)
	}
	storage = []float32{1, 2, 3, 4}
	if err := EulerStepInto(storage[1:3], []float32{1, 1}, storage[:2], sigma, sigmaNext); err == nil || !reflect.DeepEqual(storage, []float32{1, 2, 3, 4}) {
		t.Fatalf("accepted partial velocity overlap: %v err=%v", storage, err)
	}
	storage = []float32{1, 2, 3, 4}
	if err := EulerStepInto(storage[:2], storage[:2], storage[1:3], sigma, sigmaNext); err == nil || !reflect.DeepEqual(storage, []float32{1, 2, 3, 4}) {
		t.Fatalf("accepted in-place partial input overlap: %v err=%v", storage, err)
	}
	if got := testing.AllocsPerRun(100, func() {
		if err := EulerStepInto(dst, sample, []float32{2, 2}, sigma, sigmaNext); err != nil {
			t.Fatal(err)
		}
	}); got != 0 {
		t.Fatalf("EulerStepInto allocs/call=%v want 0", got)
	}
}
