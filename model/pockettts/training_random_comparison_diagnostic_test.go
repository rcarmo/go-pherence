package pockettts

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/big"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"testing"
)

const randomOracleArchive = "/workspace/exports/pocket-mpfr-oracle-20261001/oracle.json"

type randomOracleRecord struct {
	Kind, Name string
	Precision  int
	A, B       string
	Normal     string `json:"normal_bits"`
	Low        string `json:"ideal_lo"`
	High       string `json:"ideal_hi"`
	Ideal      string `json:"ideal_f32"`
	StepValue  string `json:"step_value"`
	Step       string `json:"step_f32"`
}
type randomOracleDocument struct {
	Schema        int
	MPFR, GMP     string
	Ideal, Step   string
	Cases         []randomOracleRecord
	Pairs         [][2]string `json:"pair_inputs"`
	PairReference string      `json:"pair_reference"`
}
type randomOracleInput struct {
	kind, name string
	a, b       uint64
	normal     uint32
}

// Independent literal input inventory from the reviewed frozen oracle source.
// No candidate-derived expected output or numerical acceptance threshold.
func randomComparisonInputs() []randomOracleInput {
	return []randomOracleInput{
		{"bm", "radial_min_angle_zero", 0, 0, 0},
		{"bm", "radial_max_angle_zero", math.MaxUint64, 0, 0},
		{"bm", "grid_low_bits", 4095, 2047, 0},
		{"bm", "radial_min_quarter", 0, 1 << 62, 0},
		{"bm", "radial_min_half", 0, 1 << 63, 0},
		{"bm", "radial_min_threequarter", 0, 3 << 62, 0},
		{"bm", "radial_min_angle_max", 0, math.MaxUint64, 0},
		{"bm", "radial_max_angle_max", math.MaxUint64, math.MaxUint64, 0},
		{"bm", "midradial_zero", 1 << 63, 0, 0},
		{"bm", "midradial_quarter_minus", 1 << 63, (1 << 62) - 2048, 0},
		{"bm", "midradial_quarter_plus", 1 << 63, (1 << 62) + 2048, 0},
		{"bm", "midradial_threequarter", 1 << 63, 3 << 62, 0},
		{"bm", "radial_quarter_angle_eighth", 1 << 62, 1 << 61, 0},
		{"bm", "radial_threequarter_angle_fiveeighth", 3 << 62, 5 << 61, 0},
		{"bm", "explicit_word_a", 0x123456789abcdef0, 0xfedcba9876543210, 0},
		{"bm", "explicit_word_b", 0x1111111111111111, 0x9999999999999999, 0},
		{"time", "plus_zero", 0, 0, 0}, {"time", "minus_zero", 0, 0, 0x80000000},
		{"time", "plus_one", 0, 0, 0x3f800000}, {"time", "minus_one", 0, 0, 0xbf800000},
		{"time", "plus_two", 0, 0, 0x40000000}, {"time", "minus_two", 0, 0, 0xc0000000},
		{"time", "cancel_mean", 0, 0, 0xbecccccd},
		{"time", "cancel_mean_below", 0, 0, 0xbeccccce}, {"time", "cancel_mean_above", 0, 0, 0xbecccccc},
		{"time", "plus_min_subnormal", 0, 0, 1}, {"time", "minus_min_subnormal", 0, 0, 0x80000001},
		{"time", "plus_ten", 0, 0, 0x41200000}, {"time", "minus_ten", 0, 0, 0xc1200000},
		{"time", "plus_sixteen", 0, 0, 0x41800000}, {"time", "minus_sixteen", 0, 0, 0xc1800000},
		{"time", "plus_twenty", 0, 0, 0x41a00000}, {"time", "minus_twenty", 0, 0, 0xc1a00000},
		{"time", "minus_eightyseven", 0, 0, 0xc2ae0000},
		{"time", "minus_hundred", 0, 0, 0xc2c80000}, {"time", "minus_hundredfour", 0, 0, 0xc2d00000},
		{"time", "plus_hundredtwenty", 0, 0, 0x42f00000}, {"time", "minus_hundredtwenty", 0, 0, 0xc2f00000},
		{"time", "plus_half", 0, 0, 0x3f000000},
	}
}

// Exact dyadic parser. Exponent/digit ceilings bound allocation. No lossy
// hex->decimal->float chain is used for enclosure ordering or distance.
func comparisonDyadic(text string) (*big.Rat, bool, error) {
	if len(text) == 0 || len(text) > 160 {
		return nil, false, fmt.Errorf("hex length")
	}
	negative := strings.HasPrefix(text, "-")
	body := strings.TrimPrefix(text, "-")
	parts := strings.Split(body, "p")
	if len(parts) != 2 || !strings.HasPrefix(parts[0], "0x") {
		return nil, false, fmt.Errorf("hex format")
	}
	exponent, err := strconv.Atoi(parts[1])
	if err != nil || exponent < -2048 || exponent > 2048 {
		return nil, false, fmt.Errorf("hex exponent")
	}
	mantissa := strings.TrimPrefix(parts[0], "0x")
	digits := strings.Split(mantissa, ".")
	if len(digits) > 2 || len(digits[0]) == 0 || (len(digits) == 2 && len(digits[1]) == 0) {
		return nil, false, fmt.Errorf("hex significand")
	}
	joined := strings.Join(digits, "")
	if len(joined) > 80 {
		return nil, false, fmt.Errorf("hex significand ceiling")
	}
	for _, c := range joined {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return nil, false, fmt.Errorf("hex digit")
		}
	}
	n, ok := new(big.Int).SetString(joined, 16)
	if !ok {
		return nil, false, fmt.Errorf("hex integer")
	}
	if negative {
		n.Neg(n)
	}
	if len(digits) == 2 {
		exponent -= 4 * len(digits[1])
	}
	d := big.NewInt(1)
	if exponent >= 0 {
		n.Lsh(n, uint(exponent))
	} else {
		d.Lsh(d, uint(-exponent))
	}
	return new(big.Rat).SetFrac(n, d), negative && n.Sign() == 0, nil
}
func comparisonFloatBits(text string) (uint32, error) {
	if len(text) != 8 {
		return 0, fmt.Errorf("F32 encoding")
	}
	v, err := strconv.ParseUint(text, 16, 32)
	bits := uint32(v)
	if err != nil || bits&0x7f800000 == 0x7f800000 {
		return 0, fmt.Errorf("nonfinite F32 encoding")
	}
	return bits, nil
}
func comparisonRoundBits(r *big.Rat, negativeZero bool) uint32 {
	// Dyadic numerator<=320 bits; precision1024 therefore stores it exactly.
	f := new(big.Float).SetPrec(1024).SetMode(big.ToNearestEven).SetRat(r)
	v, _ := f.Float32()
	if r.Sign() == 0 && negativeZero {
		return 0x80000000
	}
	return math.Float32bits(v)
}

type checkedRandomRecord struct {
	record              randomOracleRecord
	low, high, step     *big.Rat
	idealBits, stepBits uint32
}

func checkComparisonReference(t *testing.T) (map[string]checkedRandomRecord, [][2]string) {
	t.Helper()
	fd, err := syscall.Open(randomOracleArchive, syscall.O_RDONLY|syscall.O_NOFOLLOW|syscall.O_NONBLOCK|syscall.O_CLOEXEC, 0)
	if err != nil {
		t.Fatal(err)
	}
	file := os.NewFile(uintptr(fd), randomOracleArchive)
	info, statErr := file.Stat()
	if statErr != nil || !info.Mode().IsRegular() || info.Size() != 28464 {
		_ = file.Close()
		t.Fatal("reference regular-file/size admission")
	}
	data, readErr := io.ReadAll(io.LimitReader(file, (128<<10)+1))
	closeErr := file.Close()
	if readErr != nil || closeErr != nil || len(data) != 28464 {
		t.Fatal("reference bounded read")
	}
	// This hash is archived independent reference/input provenance, never a
	// candidate-output equality or numerical acceptance gate.
	digest := sha256.Sum256(data)
	if hex.EncodeToString(digest[:]) != "fa4df1d4ee3e67b04a0c28c72c954f1cd39d58f5714b51a8b6330f8803853b34" {
		t.Fatal("reference provenance")
	}
	var doc randomOracleDocument
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&doc); err != nil {
		t.Fatal(err)
	}
	if doc.Schema != 1 || doc.MPFR != "4.2.1" || doc.GMP != "6.3.0" || len(doc.Cases) != 78 || doc.Ideal != "exact dyadic inputs/fixed F64 tau, unrounded formula" || doc.Step != "MPFR correctly-rounded mathematical IEEE steps, NOT Go libm assumption" || doc.PairReference != "F32 affine logit is monotone; order matching scalar references, preserve equal outputs" {
		t.Fatal("reference schema/targets")
	}
	pairs := [][2]string{{"plus_zero", "plus_zero"}, {"minus_one", "plus_one"}, {"plus_one", "minus_one"}, {"minus_hundredtwenty", "plus_hundredtwenty"}}
	if !reflect.DeepEqual(doc.Pairs, pairs) {
		t.Fatal("pair corpus")
	}
	inputs := randomComparisonInputs()
	if len(inputs) != 39 {
		t.Fatal("expected fixed39 inputs")
	}
	expected := make(map[string]randomOracleInput, 39)
	for _, in := range inputs {
		expected[in.kind+":"+in.name] = in
	}
	checked := make(map[string]checkedRandomRecord, 78)
	for _, r := range doc.Cases {
		in, exists := expected[r.Kind+":"+r.Name]
		if !exists || (r.Precision != 192 && r.Precision != 256) || r.A != fmt.Sprintf("%016x", in.a) || r.B != fmt.Sprintf("%016x", in.b) || r.Normal != fmt.Sprintf("%08x", in.normal) {
			t.Fatal("reference input/precision identity")
		}
		key := fmt.Sprintf("%d:%s:%s", r.Precision, r.Kind, r.Name)
		if _, dup := checked[key]; dup {
			t.Fatal("duplicate record")
		}
		lo, lz, e1 := comparisonDyadic(r.Low)
		hi, hz, e2 := comparisonDyadic(r.High)
		step, sz, e3 := comparisonDyadic(r.StepValue)
		ib, e4 := comparisonFloatBits(r.Ideal)
		sb, e5 := comparisonFloatBits(r.Step)
		if e1 != nil || e2 != nil || e3 != nil || e4 != nil || e5 != nil {
			t.Fatal("reference finite dyadic encodings", e1, e2, e3, e4, e5)
		}
		if lo.Cmp(hi) > 0 || comparisonRoundBits(lo, lz) != ib || comparisonRoundBits(hi, hz) != ib || comparisonRoundBits(step, sz) != sb {
			t.Fatal("enclosure rounding identity")
		}
		checked[key] = checkedRandomRecord{r, lo, hi, step, ib, sb}
	}
	for _, in := range inputs {
		a, okA := checked["192:"+in.kind+":"+in.name]
		b, okB := checked["256:"+in.kind+":"+in.name]
		if !okA || !okB || a.idealBits != b.idealBits || a.stepBits != b.stepBits || a.step.Cmp(b.step) != 0 {
			t.Fatal("precision/corpus completeness")
		}
		if b.low.Cmp(a.high) > 0 || b.high.Cmp(a.low) < 0 {
			t.Fatal("precision enclosures are disjoint")
		}
	}
	return checked, pairs
}

func comparisonOrderedBits(b uint32) uint32 {
	if b&0x80000000 != 0 {
		return ^b
	}
	return b | 0x80000000
}
func comparisonULP(a, b uint32) uint64 {
	// Signed zeros coalesce for numerical ULP; sign mismatch is separately
	// reported. Remove the artificial step between -0/+0 across sign changes.
	if a&0x7fffffff == 0 && b&0x7fffffff == 0 {
		return 0
	}
	x, y := comparisonOrderedBits(a), comparisonOrderedBits(b)
	var n uint64
	if x > y {
		n = uint64(x - y)
	} else {
		n = uint64(y - x)
	}
	if (a^b)&0x80000000 != 0 && n > 0 {
		n--
	}
	return n
}

type comparisonMetric struct {
	Name, Label                      string
	Signed, Absolute, ScaledRelative float64
	ULP                              uint64
	SignedZeroMismatch               bool
}

func comparisonMetricFor(name string, candidate float32, refBits uint32) comparisonMetric {
	ref := math.Float32frombits(refBits)
	cb := math.Float32bits(candidate)
	r := float64(ref)
	delta := float64(candidate) - r
	label := "ordinary"
	if refBits&0x7fffffff == 0 {
		label = "zero_or_underflow"
	} else if refBits&0x7f800000 == 0 {
		label = "subnormal"
	} else if math.Abs(r) < 1e-6 {
		label = "near_zero"
	}
	// 2^-126 is a scale floor, not an acceptance tolerance. Near-zero and
	// underflow rows retain their own labels; raw-relative inference is unsafe.
	return comparisonMetric{name, label, delta, math.Abs(delta), math.Abs(delta) / math.Max(math.Abs(r), 0x1p-126), comparisonULP(cb, refBits), cb != refBits && cb&0x7fffffff == 0 && refBits&0x7fffffff == 0}
}
func comparisonRat(x float32) *big.Rat { return new(big.Rat).SetFloat64(float64(x)) }
func comparisonLogJSON(t *testing.T, kind string, v any) {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%s %s", kind, data)
}
func comparisonSummary(t *testing.T, key string, rows []comparisonMetric) {
	t.Helper()
	signed := make([]float64, len(rows))
	abs := make([]float64, len(rows))
	rel := make([]float64, len(rows))
	ulps := make([]uint64, len(rows))
	sum := 0.0
	worst, worstULP, worstRelative := rows[0], rows[0], rows[0]
	zeroMismatch := 0
	for i, r := range rows {
		signed[i] = r.Signed
		abs[i] = r.Absolute
		rel[i] = r.ScaledRelative
		ulps[i] = r.ULP
		sum += r.Absolute * r.Absolute
		if r.Absolute > worst.Absolute {
			worst = r
		}
		if r.ULP > worstULP.ULP {
			worstULP = r
		}
		if r.ScaledRelative > worstRelative.ScaledRelative {
			worstRelative = r
		}
		if r.SignedZeroMismatch {
			zeroMismatch++
		}
	}
	sort.Float64s(signed)
	sort.Float64s(abs)
	sort.Float64s(rel)
	sort.Slice(ulps, func(i, j int) bool { return ulps[i] < ulps[j] })
	q := func(p float64) int { return int(math.Ceil(p*float64(len(rows)))) - 1 }
	comparisonLogJSON(t, "SUMMARY", map[string]any{"family_target": key, "count": len(rows), "min_signed": signed[0], "max_signed": signed[len(rows)-1], "signed_q50": signed[q(.5)], "max_abs": abs[len(rows)-1], "rms_abs": math.Sqrt(sum / float64(len(rows))), "abs_q50": abs[q(.5)], "abs_q90": abs[q(.9)], "abs_q95": abs[q(.95)], "abs_q99": abs[q(.99)], "max_scaled_relative": rel[len(rows)-1], "scaled_relative_q95": rel[q(.95)], "max_ulp": ulps[len(rows)-1], "ulp_q95": ulps[q(.95)], "worst_abs": worst, "worst_ulp": worstULP, "worst_scaled_relative": worstRelative, "signed_zero_mismatches": zeroMismatch, "numerical_acceptance": false})
}

func checkComparisonMetrics(t *testing.T) {
	t.Helper()
	for _, c := range []struct {
		text                   string
		numerator, denominator int64
		negativeZero           bool
	}{
		{"0x1p-149", 1, 0, false},
		{"-0x0p+0", 0, 1, true},
		{"0x1.8p+1", 3, 1, false},
		{"-0x1p-1", -1, 2, false},
	} {
		r, zero, err := comparisonDyadic(c.text)
		want := new(big.Rat)
		if c.denominator == 0 {
			want.SetFrac(big.NewInt(1), new(big.Int).Lsh(big.NewInt(1), 149))
		} else {
			want.SetFrac64(c.numerator, c.denominator)
		}
		if err != nil || r.Cmp(want) != 0 || zero != c.negativeZero {
			t.Fatal("literal dyadic parser invariant")
		}
	}
	for _, c := range []struct {
		a, b uint32
		want uint64
	}{
		{0, 0x80000000, 0}, {0x80000001, 1, 2}, {0x80000001, 0, 1},
		{0x3f800000, 0x3f800001, 1}, {0xbf800000, 0xbf800001, 1},
	} {
		if comparisonULP(c.a, c.b) != c.want || comparisonULP(c.b, c.a) != c.want {
			t.Fatal("literal ULP ordering invariant")
		}
	}
	// Independent exact conversion tie checks; no transcendental expectation.
	if comparisonRoundBits(new(big.Rat).SetFrac(big.NewInt(1), new(big.Int).Lsh(big.NewInt(1), 150)), false) != 0 || comparisonRoundBits(new(big.Rat).SetFrac64(16777217, 16777216), false) != 0x3f800000 {
		t.Fatal("nearest-even conversion invariant")
	}
}

func TestTrainingRandomMPFRComparisonDiagnostic(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_RANDOM_COMPARE") != "1" {
		t.Skip("requires separately reviewed fixed-corpus error-characterisation gate")
	}
	// Independent exact metric/parser checks, then complete corpus admission
	// BEFORE invoking any candidate; no oracle is rerun.
	checkComparisonMetrics(t)
	refs, pairs := checkComparisonReference(t)
	groups := map[string][]comparisonMetric{}
	for _, in := range randomComparisonInputs() {
		r := refs["256:"+in.kind+":"+in.name]
		var got float32
		var err error
		if in.kind == "bm" {
			got, err = trainingNormalFromRaw(in.a, in.b)
		} else {
			got, err = trainingTimeFromNormal(math.Float32frombits(in.normal))
		}
		if err != nil || !isFinite(got) || (in.kind == "time" && (got < 0 || got > 1)) {
			t.Fatal("candidate domain invariant", in.name, err)
		}
		ideal := comparisonMetricFor(in.name, got, r.idealBits)
		step := comparisonMetricFor(in.name, got, r.stepBits)
		groups[in.kind+"/rounded_ideal"] = append(groups[in.kind+"/rounded_ideal"], ideal)
		groups[in.kind+"/rounded_step"] = append(groups[in.kind+"/rounded_step"], step)
		groups[in.kind+"/rounded_ideal/"+ideal.Label] = append(groups[in.kind+"/rounded_ideal/"+ideal.Label], ideal)
		groups[in.kind+"/rounded_step/"+step.Label] = append(groups[in.kind+"/rounded_step/"+step.Label], step)
		cr := comparisonRat(got)
		distance := new(big.Rat)
		if cr.Cmp(r.low) < 0 {
			distance.Sub(cr, r.low)
		} else if cr.Cmp(r.high) > 0 {
			distance.Sub(cr, r.high)
		}
		stepDistance := new(big.Rat).Sub(cr, r.step)
		comparisonLogJSON(t, "CASE", map[string]any{"family": in.kind, "name": in.name, "candidate_bits": fmt.Sprintf("%08x", math.Float32bits(got)), "candidate": got, "rounded_ideal": ideal, "rounded_step": step, "ideal_lo": r.record.Low, "ideal_hi": r.record.High, "step_value": r.record.StepValue, "signed_distance_to_ideal_interval_exact": distance.RatString(), "signed_error_to_unrounded_step_exact": stepDistance.RatString(), "distance_role": "exact dyadic arithmetic; no numerical acceptance threshold"})
	}
	keys := make([]string, 0, len(groups))
	for key := range groups {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		comparisonSummary(t, key, groups[key])
	}
	for _, pair := range pairs {
		a, b := refs["256:time:"+pair[0]], refs["256:time:"+pair[1]]
		ab, _ := strconv.ParseUint(a.record.Normal, 16, 32)
		bb, _ := strconv.ParseUint(b.record.Normal, 16, 32)
		s, end, err := trainingPairFromNormals(math.Float32frombits(uint32(ab)), math.Float32frombits(uint32(bb)))
		if err != nil || !isFinite(s) || !isFinite(end) || s < 0 || s > end || end > 1 || (pair[0] == pair[1] && s != end) {
			t.Fatal("pair domain/equality invariant", pair, err)
		}
		low, high := a, b
		if math.Float32frombits(uint32(ab)) > math.Float32frombits(uint32(bb)) {
			low, high = b, a
		}
		comparisonLogJSON(t, "PAIR", map[string]any{"inputs": pair, "s_bits": fmt.Sprintf("%08x", math.Float32bits(s)), "t_bits": fmt.Sprintf("%08x", math.Float32bits(end)), "s_rounded_ideal": comparisonMetricFor(pair[0], s, low.idealBits), "t_rounded_ideal": comparisonMetricFor(pair[1], end, high.idealBits), "s_rounded_step": comparisonMetricFor(pair[0], s, low.stepBits), "t_rounded_step": comparisonMetricFor(pair[1], end, high.stepBits), "numerical_acceptance": false})
	}
}
