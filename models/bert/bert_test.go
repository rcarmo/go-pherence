package bert

import (
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/tensor"
)

func TestLoadGTESmall(t *testing.T) {
	path := gteSmallPath(t)

	m, err := LoadGTESmall(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	if m.Config.NumLayers != 12 {
		t.Fatalf("layers=%d want 12", m.Config.NumLayers)
	}
	if m.Config.HiddenSize != 384 {
		t.Fatalf("hidden=%d want 384", m.Config.HiddenSize)
	}
	t.Logf("Loaded GTE-small: %d layers, hidden=%d", m.Config.NumLayers, m.Config.HiddenSize)
}

func TestForwardGTESmall(t *testing.T) {
	path := gteSmallPath(t)

	m, err := LoadGTESmall(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	// Tokenize "I love cats" manually (CLS=101, I=1045, love=2293, cats=8870, SEP=102)
	tokenIDs := []int{101, 1045, 2293, 8870, 102}
	attnMask := []bool{true, true, true, true, true}

	emb := m.Embed(tokenIDs, attnMask)
	if len(emb) != 384 {
		t.Fatalf("embedding dim=%d want 384", len(emb))
	}

	// Check L2 normalized
	norm := float32(0)
	for _, v := range emb {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if math.Abs(float64(norm-1.0)) > 0.01 {
		t.Fatalf("norm=%v want ~1.0", norm)
	}

	// Check not all zeros
	nonZero := 0
	for _, v := range emb {
		if v != 0 {
			nonZero++
		}
	}
	if nonZero < 100 {
		t.Fatalf("too many zeros: %d non-zero out of 384", nonZero)
	}

	t.Logf("Embedding[0:5]: %v", emb[:5])
	t.Logf("Norm: %v, non-zero: %d/384", norm, nonZero)
}

func TestCheckedMulInt(t *testing.T) {
	if got, ok := checkedMulInt(4, 7); !ok || got != 28 {
		t.Fatalf("checkedMulInt(4,7)=%d,%v want 28,true", got, ok)
	}
	if _, ok := checkedMulInt(-1, 7); ok {
		t.Fatal("checkedMulInt accepted negative lhs")
	}
	maxInt := int(^uint(0) >> 1)
	if _, ok := checkedMulInt(maxInt/2+1, 3); ok {
		t.Fatal("checkedMulInt accepted overflow")
	}
}

func TestMHAInPlaceMatchesTensorAttention(t *testing.T) {
	seqLen, heads, headDim := 2, 1, 2
	q := []float32{1, 0, 0, 1}
	k := []float32{1, 0, 0, 1}
	v := []float32{1, 2, 3, 4}
	want := multiHeadAttention(
		tensor.FromFloat32(q, []int{seqLen, heads * headDim}),
		tensor.FromFloat32(k, []int{seqLen, heads * headDim}),
		tensor.FromFloat32(v, []int{seqLen, heads * headDim}),
		seqLen, heads, headDim,
	).Data()
	got := make([]float32, len(want))
	mhaInPlace(got, q, k, v, make([]float32, heads*seqLen*seqLen), seqLen, heads, headDim)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("got[%d]=%g want %g", i, got[i], want[i])
		}
	}
}

func TestMHAInPlaceRejectsMalformedInputs(t *testing.T) {
	out := []float32{99, 100, 101, 102}
	q := []float32{1, 0, 0, 1}
	k := []float32{1, 0, 0, 1}
	v := []float32{1, 2, 3, 4}
	scores := make([]float32, 4)
	mhaInPlace(out, q, k, v, scores, 2, 1, 2)
	if out[0] == 99 && out[1] == 100 && out[2] == 101 && out[3] == 102 {
		t.Fatal("mhaInPlace did not write valid output")
	}
	out = []float32{99, 100, 101, 102}
	mhaInPlace(out[:3], q, k, v, scores, 2, 1, 2)
	if out[0] != 99 || out[3] != 102 {
		t.Fatalf("malformed mhaInPlace mutated output: %v", out)
	}
	mhaInPlace(out, q, k, v, scores[:3], 2, 1, 2)
	if out[0] != 99 || out[3] != 102 {
		t.Fatalf("short scores mhaInPlace mutated output: %v", out)
	}
}

func TestLinearInPlaceUsesCheckedSIMD(t *testing.T) {
	out := make([]float32, 7)
	out[6] = 123
	x := []float32{1, 2, 3, 4}
	w := []float32{1, 0, 0, 1, 2, 1}
	bias := []float32{0.5, -1, 2}
	linearInPlace(out[:6], x, w, bias, 2, 2, 3)
	want := []float32{3.5, 3, 4, 7.5, 7, 6}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("out[%d]=%g want %g", i, out[i], want[i])
		}
	}
	if out[6] != 123 {
		t.Fatal("linearInPlace mutated tail")
	}
	linearInPlace(out[:1], x, w, bias, 2, 2, 3)
	if out[6] != 123 {
		t.Fatal("malformed linearInPlace mutated tail")
	}
}

func gteSmallPath(tb testing.TB) string {
	tb.Helper()
	if path := os.Getenv("SAFETENSORS_PATH"); path != "" {
		if _, err := os.Stat(path); err == nil {
			return path
		}
		tb.Skipf("model not found: %s", path)
	}
	for _, path := range []string{
		"../../../gte-go/models/gte-small/model.safetensors",
		"../../gte-go/models/gte-small/model.safetensors",
		"../gte-go/models/gte-small/model.safetensors",
	} {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	tb.Skip("GTE-small safetensors fixture not found")
	return ""
}

func BenchmarkGTESmallEmbed(b *testing.B) {
	path := gteSmallPath(b)
	m, err := LoadGTESmall(path)
	if err != nil {
		b.Fatalf("load: %v", err)
	}
	tokenIDs := []int{101, 1045, 2293, 8870, 102} // "I love cats"
	attnMask := []bool{true, true, true, true, true}

	// Warmup
	for i := 0; i < 3; i++ {
		_ = m.Embed(tokenIDs, attnMask)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = m.Embed(tokenIDs, attnMask)
	}
}

func TestForwardFastCorrectness(t *testing.T) {
	path := gteSmallPath(t)
	m, err := LoadGTESmall(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	tokenIDs := []int{101, 1045, 2293, 8870, 102}
	attnMask := []bool{true, true, true, true, true}

	slow := m.Embed(tokenIDs, attnMask)
	fast := m.EmbedFast(tokenIDs, attnMask)

	maxDiff := float32(0)
	for i := range slow {
		d := slow[i] - fast[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 0.001 {
		t.Fatalf("fast vs slow maxDiff=%v (too large)", maxDiff)
	}
	t.Logf("fast vs slow maxDiff=%v", maxDiff)
}

func BenchmarkGTESmallEmbedFast(b *testing.B) {
	path := gteSmallPath(b)
	m, err := LoadGTESmall(path)
	if err != nil {
		b.Fatalf("load: %v", err)
	}
	tokenIDs := []int{101, 1045, 2293, 8870, 102}
	attnMask := []bool{true, true, true, true, true}
	for i := 0; i < 3; i++ {
		_ = m.EmbedFast(tokenIDs, attnMask)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = m.EmbedFast(tokenIDs, attnMask)
	}
}
