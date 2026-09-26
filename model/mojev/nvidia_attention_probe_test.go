package mojev

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"testing"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/nvidia/ptx"
	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

const (
	moJevAttentionProbeGate   = "GO_PHERENCE_MOJEV_ATTENTION_PROBE"
	moJevAttentionProbePrefix = "GO_PHERENCE_MOJEV_ATTENTION_PROBE_"

	moJevAttentionProbeHeads     = 8
	moJevAttentionProbeKVHeads   = 2
	moJevAttentionProbeHeadDim   = 256
	moJevAttentionProbeQGWidth   = 4096
	moJevAttentionProbeKVWidth   = 512
	moJevAttentionProbeOutWidth  = 2048
	moJevAttentionProbeBlockSize = 256
)

type moJevAttentionProbeConfig struct {
	kernel   string
	rows     int
	state    int
	question int
	large    bool
}

func (c moJevAttentionProbeConfig) actualKernel() string {
	if c.kernel == "warp" {
		return "mj_tree_attention_warp"
	}
	if c.kernel == "tree" {
		return "mj_tree_attention"
	}
	return "mj_attention"
}

func (c moJevAttentionProbeConfig) prefix() int { return c.state + c.question }
func (c moJevAttentionProbeConfig) gridX() uint32 {
	if c.kernel == "warp" {
		return uint32(c.rows)
	} // eight head/query warps per block
	return uint32(c.rows * moJevAttentionProbeHeads)
}

type moJevAttentionProbeLookup func(string) (string, bool)

func moJevAttentionProbeEnabled(lookup moJevAttentionProbeLookup) bool {
	v, ok := lookup(moJevAttentionProbeGate)
	return ok && v == "1"
}

func parseMoJevAttentionProbeConfig(lookup moJevAttentionProbeLookup) (moJevAttentionProbeConfig, error) {
	cfg := moJevAttentionProbeConfig{kernel: "compact", rows: 3, state: 1, question: 1}
	if v, ok := lookup(moJevAttentionProbePrefix + "KERNEL"); ok {
		switch v {
		case "compact", "tree", "warp":
			cfg.kernel = v
		default:
			return cfg, fmt.Errorf("%sKERNEL=%q: want compact, tree or warp", moJevAttentionProbePrefix, v)
		}
	}
	var err error
	if cfg.rows, err = parseMoJevAttentionProbeInt(lookup, moJevAttentionProbePrefix+"ROWS", cfg.rows); err != nil {
		return cfg, err
	}
	if cfg.state, err = parseMoJevAttentionProbeInt(lookup, moJevAttentionProbePrefix+"STATE", cfg.state); err != nil {
		return cfg, err
	}
	if cfg.question, err = parseMoJevAttentionProbeInt(lookup, moJevAttentionProbePrefix+"QUESTION", cfg.question); err != nil {
		return cfg, err
	}
	if cfg.large, err = parseMoJevAttentionProbeBool01(lookup, moJevAttentionProbePrefix+"LARGE"); err != nil {
		return cfg, err
	}
	if err := validateMoJevAttentionProbeConfig(cfg); err != nil {
		return cfg, err
	}
	return cfg, nil
}

func parseMoJevAttentionProbeInt(lookup moJevAttentionProbeLookup, key string, def int) (int, error) {
	v, ok := lookup(key)
	if !ok {
		return def, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0, fmt.Errorf("%s=%q: %w", key, v, err)
	}
	return n, nil
}

func parseMoJevAttentionProbeBool01(lookup moJevAttentionProbeLookup, key string) (bool, error) {
	v, ok := lookup(key)
	if !ok {
		return false, nil
	}
	switch v {
	case "0":
		return false, nil
	case "1":
		return true, nil
	default:
		return false, fmt.Errorf("%s=%q: want 0 or 1", key, v)
	}
}

func validateMoJevAttentionProbeConfig(cfg moJevAttentionProbeConfig) error {
	if cfg.kernel != "compact" && cfg.kernel != "tree" && cfg.kernel != "warp" {
		return fmt.Errorf("invalid attention probe kernel %q", cfg.kernel)
	}
	if cfg.rows < 3 || cfg.rows > 512 {
		return fmt.Errorf("attention probe rows=%d outside [3,512]", cfg.rows)
	}
	if cfg.state < 1 || cfg.state >= cfg.rows {
		return fmt.Errorf("attention probe state=%d outside [1,rows)", cfg.state)
	}
	if cfg.question < 1 || cfg.question >= cfg.rows {
		return fmt.Errorf("attention probe question=%d outside [1,rows)", cfg.question)
	}
	// Use a bounded subtraction so malformed host integers cannot overflow
	// before configuration validation, CPU reference work or device access.
	if cfg.question >= cfg.rows-cfg.state {
		return fmt.Errorf("attention probe state=%d question=%d must leave a candidate within rows=%d", cfg.state, cfg.question, cfg.rows)
	}
	if cfg.rows > 33 && !cfg.large {
		return fmt.Errorf("attention probe rows=%d requires %sLARGE=1", cfg.rows, moJevAttentionProbePrefix)
	}
	return nil
}

func moJevAttentionProbeLookupMap(m map[string]string) moJevAttentionProbeLookup {
	return func(key string) (string, bool) {
		v, ok := m[key]
		return v, ok
	}
}

func makeMoJevAttentionProbeInputs(rows int) ([]float32, []float32, []float32) {
	qg := make([]float32, rows*moJevAttentionProbeQGWidth)
	k := make([]float32, rows*moJevAttentionProbeKVWidth)
	v := make([]float32, rows*moJevAttentionProbeKVWidth)
	for row := 0; row < rows; row++ {
		for head := 0; head < moJevAttentionProbeHeads; head++ {
			base := row*moJevAttentionProbeQGWidth + head*moJevAttentionProbeHeadDim*2
			for d := 0; d < moJevAttentionProbeHeadDim; d++ {
				qg[base+d] = float32(((row*17+head*11+d*3)%29)-14) * 0.015
				qg[base+moJevAttentionProbeHeadDim+d] = float32(((row*7+head*13+d*5)%23)-11) * 0.09
			}
		}
		for kvh := 0; kvh < moJevAttentionProbeKVHeads; kvh++ {
			base := row*moJevAttentionProbeKVWidth + kvh*moJevAttentionProbeHeadDim
			for d := 0; d < moJevAttentionProbeHeadDim; d++ {
				k[base+d] = float32(((row*19+kvh*7+d*2)%31)-15) * 0.012
				v[base+d] = float32(((row*23+kvh*5+d*7)%37)-18) * 0.02
			}
		}
	}
	return qg, k, v
}

func moJevAttentionProbeVisibleRows(cfg moJevAttentionProbeConfig, row int) int {
	if row < cfg.state {
		return cfg.state
	}
	if row < cfg.prefix() {
		return cfg.prefix()
	}
	return cfg.rows
}

func moJevAttentionProbeOracle(cfg moJevAttentionProbeConfig, qg, k, v []float32) ([]float32, error) {
	if err := validateMoJevAttentionProbeConfig(cfg); err != nil {
		return nil, err
	}
	if len(qg) != cfg.rows*moJevAttentionProbeQGWidth || len(k) != cfg.rows*moJevAttentionProbeKVWidth || len(v) != cfg.rows*moJevAttentionProbeKVWidth {
		return nil, fmt.Errorf("attention probe oracle: invalid input lengths qg=%d k=%d v=%d", len(qg), len(k), len(v))
	}
	out := make([]float32, cfg.rows*moJevAttentionProbeOutWidth)
	scores := make([]float64, cfg.rows)
	for row := 0; row < cfg.rows; row++ {
		visible := moJevAttentionProbeVisibleRows(cfg, row)
		for head := 0; head < moJevAttentionProbeHeads; head++ {
			kvh := head / (moJevAttentionProbeHeads / moJevAttentionProbeKVHeads)
			qBase := row*moJevAttentionProbeQGWidth + head*moJevAttentionProbeHeadDim*2
			kHeadOffset := kvh * moJevAttentionProbeHeadDim
			maxScore := math.Inf(-1)
			for src := 0; src < visible; src++ {
				dot := 0.0
				kBase := src*moJevAttentionProbeKVWidth + kHeadOffset
				for d := 0; d < moJevAttentionProbeHeadDim; d++ {
					dot += float64(qg[qBase+d]) * float64(k[kBase+d])
				}
				scores[src] = dot * 0.0625
				if scores[src] > maxScore {
					maxScore = scores[src]
				}
			}
			sum := 0.0
			for src := 0; src < visible; src++ {
				scores[src] = math.Exp(scores[src] - maxScore)
				sum += scores[src]
			}
			outBase := row*moJevAttentionProbeOutWidth + head*moJevAttentionProbeHeadDim
			for d := 0; d < moJevAttentionProbeHeadDim; d++ {
				acc := 0.0
				for src := 0; src < visible; src++ {
					vBase := src*moJevAttentionProbeKVWidth + kHeadOffset
					acc += (scores[src] / sum) * float64(v[vBase+d])
				}
				gate := float64(qg[qBase+moJevAttentionProbeHeadDim+d])
				out[outBase+d] = float32(acc * (1 / (1 + math.Exp(-gate))))
			}
		}
	}
	return out, nil
}

func fillMoJevAttentionProbeSentinel(dst []float32, value float32) {
	for i := range dst {
		dst[i] = value
	}
}

func TestMoJevAttentionProbeConfig(t *testing.T) {
	cfg, err := parseMoJevAttentionProbeConfig(moJevAttentionProbeLookupMap(nil))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.kernel != "compact" || cfg.rows != 3 || cfg.state != 1 || cfg.question != 1 || cfg.large {
		t.Fatalf("defaults = %+v", cfg)
	}
	cfg, err = parseMoJevAttentionProbeConfig(moJevAttentionProbeLookupMap(map[string]string{
		moJevAttentionProbePrefix + "KERNEL":   "tree",
		moJevAttentionProbePrefix + "ROWS":     "33",
		moJevAttentionProbePrefix + "STATE":    "10",
		moJevAttentionProbePrefix + "QUESTION": "11",
		moJevAttentionProbePrefix + "LARGE":    "1",
	}))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.kernel != "tree" || cfg.rows != 33 || cfg.state != 10 || cfg.question != 11 || !cfg.large {
		t.Fatalf("explicit = %+v", cfg)
	}
	if !moJevAttentionProbeEnabled(moJevAttentionProbeLookupMap(map[string]string{moJevAttentionProbeGate: "1"})) {
		t.Fatal("explicit gate not recognized")
	}
	if moJevAttentionProbeEnabled(moJevAttentionProbeLookupMap(map[string]string{moJevAttentionProbeGate: "0"})) {
		t.Fatal("non-1 gate accepted")
	}
	if moJevAttentionProbeEnabled(moJevAttentionProbeLookupMap(map[string]string{"GO_PHERENCE_MOJEV_NVIDIA": "1"})) {
		t.Fatal("ordinary GPU gate must not enable the isolated probe")
	}
	for _, rows := range []int{3, 10, 33, 34, 512} {
		for _, kernel := range []string{"compact", "tree", "warp"} {
			cfg, err := parseMoJevAttentionProbeConfig(moJevAttentionProbeLookupMap(map[string]string{
				moJevAttentionProbePrefix + "ROWS":   strconv.Itoa(rows),
				moJevAttentionProbePrefix + "KERNEL": kernel,
				moJevAttentionProbePrefix + "LARGE":  "1",
			}))
			if err != nil {
				t.Fatal(err)
			}
			wantKernel := "mj_attention"
			if kernel == "tree" {
				wantKernel = "mj_tree_attention"
			}
			wantGrid := uint32(rows * 8)
			if kernel == "warp" {
				wantKernel, wantGrid = "mj_tree_attention_warp", uint32(rows)
			}
			if cfg.actualKernel() != wantKernel || cfg.gridX() != wantGrid {
				t.Fatal("wrong kernel dispatch", cfg)
			}
		}
	}
}

func TestMoJevAttentionProbeConfigRejectsGeometry(t *testing.T) {
	for name, env := range map[string]map[string]string{
		"bad_kernel":                {moJevAttentionProbePrefix + "KERNEL": "invalid"},
		"rows_too_small":            {moJevAttentionProbePrefix + "ROWS": "2"},
		"rows_too_large":            {moJevAttentionProbePrefix + "ROWS": "513", moJevAttentionProbePrefix + "LARGE": "1"},
		"state_zero":                {moJevAttentionProbePrefix + "STATE": "0"},
		"state_int_overflow_sum":    {moJevAttentionProbePrefix + "STATE": strconv.Itoa(int(^uint(0) >> 1)), moJevAttentionProbePrefix + "QUESTION": "2"},
		"question_int_overflow_sum": {moJevAttentionProbePrefix + "STATE": "2", moJevAttentionProbePrefix + "QUESTION": strconv.Itoa(int(^uint(0) >> 1))},
		"rows_malformed":            {moJevAttentionProbePrefix + "ROWS": "bad"},
		"state_malformed":           {moJevAttentionProbePrefix + "STATE": "bad"},
		"question_malformed":        {moJevAttentionProbePrefix + "QUESTION": "bad"},
		"rows_parse_overflow":       {moJevAttentionProbePrefix + "ROWS": "999999999999999999999999"},
		"question_zero":             {moJevAttentionProbePrefix + "QUESTION": "0"},
		"bad_sum":                   {moJevAttentionProbePrefix + "ROWS": "7", moJevAttentionProbePrefix + "STATE": "3", moJevAttentionProbePrefix + "QUESTION": "4"},
		"large_required":            {moJevAttentionProbePrefix + "ROWS": "34"},
		"large_not_bool01":          {moJevAttentionProbePrefix + "LARGE": "yes"},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := parseMoJevAttentionProbeConfig(moJevAttentionProbeLookupMap(env)); err == nil {
				t.Fatal("accepted invalid geometry")
			}
		})
	}
}

func TestMoJevAttentionProbeOracleSanity(t *testing.T) {
	cfg := moJevAttentionProbeConfig{kernel: "compact", rows: 3, state: 1, question: 1}
	qg := make([]float32, cfg.rows*moJevAttentionProbeQGWidth)
	k := make([]float32, cfg.rows*moJevAttentionProbeKVWidth)
	v := make([]float32, cfg.rows*moJevAttentionProbeKVWidth)
	for i := range v {
		v[i] = float32(i%17-8) * 0.1
	}
	got, err := moJevAttentionProbeOracle(cfg, qg, k, v)
	if err != nil {
		t.Fatal(err)
	}
	limits := []int{1, 2, 3}
	for row, visible := range limits {
		for head := 0; head < moJevAttentionProbeHeads; head++ {
			kvh := head / (moJevAttentionProbeHeads / moJevAttentionProbeKVHeads)
			for d := 0; d < moJevAttentionProbeHeadDim; d++ {
				sum := 0.0
				for src := 0; src < visible; src++ {
					sum += float64(v[src*moJevAttentionProbeKVWidth+kvh*moJevAttentionProbeHeadDim+d])
				}
				want := float32(sum / float64(visible) * 0.5)
				idx := row*moJevAttentionProbeOutWidth + head*moJevAttentionProbeHeadDim + d
				if diff := math.Abs(float64(got[idx] - want)); diff > 1e-6 {
					t.Fatalf("row=%d head=%d dim=%d got=%g want=%g diff=%g", row, head, d, got[idx], want, diff)
				}
			}
		}
	}
	cfg.kernel = "tree"
	again, err := moJevAttentionProbeOracle(cfg, qg, k, v)
	if err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if got[i] != again[i] {
			t.Fatalf("tree oracle mismatch at %d", i)
		}
	}
}

func TestMoJevAttentionProbeOracleNonuniform(t *testing.T) {
	cfg := moJevAttentionProbeConfig{kernel: "compact", rows: 3, state: 1, question: 1}
	qg, k, v := make([]float32, 3*4096), make([]float32, 3*512), make([]float32, 3*512)
	for row := 0; row < 3; row++ {
		for head := 0; head < 8; head++ {
			qg[row*4096+head*512] = 1
		}
		for head := 0; head < 2; head++ {
			k[row*512+head*256] = float32(16 * row)
			for d := 0; d < 256; d++ {
				v[row*512+head*256+d] = float32(row + head*3)
			}
		}
	}
	got, err := moJevAttentionProbeOracle(cfg, qg, k, v)
	if err != nil {
		t.Fatal(err)
	}
	// Q.K/16 gives scores [0,1,2]; the all-zero gate gives exactly 1/2.
	// Independent closed-form prefix sums exercise nonuniform softmax, head
	// sharing and three visibility boundaries, unlike the uniform sanity case.
	for row := 0; row < 3; row++ {
		denom, weighted := 0.0, 0.0
		for src := 0; src <= row; src++ {
			w := math.Exp(float64(src))
			denom += w
			weighted += w * float64(src)
		}
		for head := 0; head < 8; head++ {
			want := float32(.5 * (weighted/denom + float64((head/4)*3)))
			for d := 0; d < 256; d++ {
				if x := got[row*2048+head*256+d]; math.Abs(float64(x-want)) > 1e-6 {
					t.Fatalf("row=%d head=%d got=%g want=%g", row, head, x, want)
				}
			}
		}
	}
	if _, err := moJevAttentionProbeOracle(cfg, qg[:1], k, v); err == nil {
		t.Fatal("short oracle inputs accepted")
	}
	cfg.rows = 2
	if _, err := moJevAttentionProbeOracle(cfg, qg, k, v); err == nil {
		t.Fatal("invalid oracle geometry accepted")
	}
}

func TestMoJevAttentionProbeInputs(t *testing.T) {
	q, k, v := makeMoJevAttentionProbeInputs(3)
	if len(q) != 3*4096 || len(k) != 3*512 || len(v) != 3*512 {
		t.Fatal("generator shape")
	}
	for _, xs := range [][]float32{q, k, v} {
		for _, x := range xs {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				t.Fatal("nonfinite fixture")
			}
		}
	}
	var guards [36]float32
	fillMoJevAttentionProbeSentinel(guards[:], 17)
	for _, x := range guards {
		if x != 17 {
			t.Fatal("sentinel initialization")
		}
	}
}

func TestMoJevPTXAttentionIsolated(t *testing.T) {
	if !moJevAttentionProbeEnabled(os.LookupEnv) {
		t.Skip("set GO_PHERENCE_MOJEV_ATTENTION_PROBE=1 to run isolated MoJev attention kernel probe")
	}
	cfg, err := parseMoJevAttentionProbeConfig(os.LookupEnv)
	if err != nil {
		t.Fatal(err)
	}
	actualKernel := cfg.actualKernel()
	var treeRows []uint32
	if cfg.kernel != "compact" {
		treeRows = make([]uint32, cfg.rows*4)
		if err := fillGPUTree(treeRows, cfg.rows, cfg.state, cfg.question, []int{cfg.rows}); err != nil {
			t.Fatal(err)
		}
	}
	qg, k, v := makeMoJevAttentionProbeInputs(cfg.rows)
	want, err := moJevAttentionProbeOracle(cfg, qg, k, v)
	if err != nil {
		t.Fatal(err)
	}
	module, err := nvidia.LoadPTXFunctions(ptx.MoJev, []string{actualKernel})
	if err != nil {
		t.Fatal(fmt.Errorf("%s module load: %w", actualKernel, err))
	}
	var buffers []*nvidia.Buffer
	defer func() {
		if err := module.Close(); err != nil {
			t.Errorf("%s cleanup module close failed; leaving %d device buffers allocated: %v", actualKernel, len(buffers), err)
			return
		}
		for i := len(buffers) - 1; i >= 0; i-- {
			buffers[i].Free()
		}
	}()
	allocUpload := func(name string, x []float32) *nvidia.Buffer {
		t.Helper()
		b, err := nvidia.Malloc(len(x))
		if err != nil {
			t.Fatal(fmt.Errorf("%s alloc %s: %w", actualKernel, name, err))
		}
		buffers = append(buffers, b)
		if err := b.Upload(x); err != nil {
			t.Fatal(fmt.Errorf("%s upload %s: %w", actualKernel, name, err))
		}
		return b
	}
	qgBuf := allocUpload("qg", qg)
	kBuf := allocUpload("k", k)
	vBuf := allocUpload("v", v)
	const (
		guardBefore = 17
		guardAfter  = 19
	)
	guardBits := uint32(0x4d51ab3f)
	guardValue := math.Float32frombits(guardBits)
	fullOut := make([]float32, guardBefore+len(want)+guardAfter)
	fillMoJevAttentionProbeSentinel(fullOut, guardValue)
	outBuf := allocUpload("out", fullOut)
	outPtr := nvidia.CUdeviceptr(uint64(outBuf.Ptr) + uint64(guardBefore*4))
	if outPtr <= outBuf.Ptr {
		t.Fatal("output pointer offset failed")
	}
	var treeBuf *nvidia.Buffer
	if cfg.kernel != "compact" {
		b, err := nvidia.Malloc(len(treeRows))
		if err != nil {
			t.Fatal(fmt.Errorf("%s alloc tree: %w", actualKernel, err))
		}
		buffers = append(buffers, b)
		if err := b.UploadUint32(treeRows); err != nil {
			t.Fatal(fmt.Errorf("%s upload tree: %w", actualKernel, err))
		}
		treeBuf = b
	}
	rows := int32(cfg.rows)
	state := int32(cfg.state)
	question := int32(cfg.question)
	prefix := int32(cfg.prefix())
	t.Logf("launch %s rows=%d ns=%d nq=%d grid=(%d,1,1) block=(%d,1,1)", actualKernel, cfg.rows, cfg.state, cfg.question, cfg.gridX(), moJevAttentionProbeBlockSize)
	switch cfg.kernel {
	case "compact":
		err = nvidia.LaunchKernel(module.Function(actualKernel), cfg.gridX(), 1, 1, moJevAttentionProbeBlockSize, 1, 1, 0,
			unsafe.Pointer(&qgBuf.Ptr), unsafe.Pointer(&kBuf.Ptr), unsafe.Pointer(&vBuf.Ptr), unsafe.Pointer(&outPtr), unsafe.Pointer(&rows), unsafe.Pointer(&state), unsafe.Pointer(&question))
	case "tree", "warp":
		err = nvidia.LaunchKernel(module.Function(actualKernel), cfg.gridX(), 1, 1, moJevAttentionProbeBlockSize, 1, 1, 0,
			unsafe.Pointer(&qgBuf.Ptr), unsafe.Pointer(&kBuf.Ptr), unsafe.Pointer(&vBuf.Ptr), unsafe.Pointer(&outPtr), unsafe.Pointer(&rows), unsafe.Pointer(&treeBuf.Ptr), unsafe.Pointer(&prefix))
	default:
		t.Fatalf("unexpected kernel %q", cfg.kernel)
	}
	if err != nil {
		t.Fatal(fmt.Errorf("%s launch: %w", actualKernel, err))
	}
	if err := nvidia.SyncErr(); err != nil {
		t.Fatal(fmt.Errorf("%s sync: %w", actualKernel, err))
	}
	gotFull := make([]float32, len(fullOut))
	if err := outBuf.Download(gotFull); err != nil {
		t.Fatal(fmt.Errorf("%s download: %w", actualKernel, err))
	}
	for i := 0; i < guardBefore; i++ {
		if math.Float32bits(gotFull[i]) != guardBits {
			t.Fatalf("%s left guard[%d] changed: got=%#x want=%#x", actualKernel, i, math.Float32bits(gotFull[i]), guardBits)
		}
	}
	for i := guardBefore + len(want); i < len(gotFull); i++ {
		if math.Float32bits(gotFull[i]) != guardBits {
			t.Fatalf("%s right guard[%d] changed: got=%#x want=%#x", actualKernel, i-(guardBefore+len(want)), math.Float32bits(gotFull[i]), guardBits)
		}
	}
	got := gotFull[guardBefore : guardBefore+len(want)]
	maxDiff := 0.0
	for i, x := range got {
		d := math.Abs(float64(x - want[i]))
		if d > maxDiff {
			maxDiff = d
		}
		// Keep the existing synthetic attention gate; this is a new diagnostic
		// fixture, not permission to loosen the established kernel check.
		const limit = 1e-6
		if math.IsNaN(d) || d > limit {
			row := i / moJevAttentionProbeOutWidth
			col := i % moJevAttentionProbeOutWidth
			t.Fatalf("%s row=%d col=%d got=%g want=%g diff=%g limit=%g", actualKernel, row, col, x, want[i], d, limit)
		}
	}
	t.Logf("%s max_diff=%g", actualKernel, maxDiff)
}
