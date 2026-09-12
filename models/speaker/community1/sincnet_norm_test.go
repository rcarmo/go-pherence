package community1

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"math/big"
	"math/rand"
	"os"
	"testing"
)

func TestSincNetFMA32ExactRounding(t *testing.T) {
	oracle := func(a, b, c float32) float32 {
		x := new(big.Float).SetPrec(1024).SetFloat64(float64(a))
		y := new(big.Float).SetPrec(1024).SetFloat64(float64(b))
		z := new(big.Float).SetPrec(1024).SetFloat64(float64(c))
		x.Mul(x, y)
		x.Add(x, z)
		result, _ := x.Float32()
		return result
	}
	rng := rand.New(rand.NewSource(9917))
	checked := 0
	for i := 0; i < 20000; i++ {
		a, b, c := math.Float32frombits(rng.Uint32()), math.Float32frombits(rng.Uint32()), math.Float32frombits(rng.Uint32())
		if math.IsNaN(float64(a)) || math.IsNaN(float64(b)) || math.IsNaN(float64(c)) || math.IsInf(float64(a), 0) || math.IsInf(float64(b), 0) || math.IsInf(float64(c), 0) {
			continue
		}
		got, want := sincNetFMA32(a, b, c), oracle(a, b, c)
		if math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("fma bits %08x*%08x+%08x got%08x want%08x", math.Float32bits(a), math.Float32bits(b), math.Float32bits(c), math.Float32bits(got), math.Float32bits(want))
		}
		checked++
	}
	// Construct a product exactly halfway between float32s; tiny c is lost in
	// float64 FMA but selects the opposite float32 endpoint in exact arithmetic.
	found := false
	for i := 0; i < 10000 && !found; i++ {
		a := math.Float32frombits(0x3f800000 | uint32(i))
		b := float32(1.5)
		product := float64(a) * float64(b)
		rounded := float32(product)
		if float64(rounded) == product {
			continue
		}
		for _, c := range []float32{math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32} {
			want := oracle(a, b, c)
			if math.Float32bits(want) != math.Float32bits(float32(math.FMA(float64(a), float64(b), float64(c)))) {
				if math.Float32bits(sincNetFMA32(a, b, c)) != math.Float32bits(want) {
					t.Fatal("midpoint correction")
				}
				found = true
				t.Logf("double-round regression a=%08x b=%08x c=%08x", math.Float32bits(a), math.Float32bits(b), math.Float32bits(c))
			}
		}
	}
	if !found {
		t.Fatal("no adversarial midpoint found")
	}
	for _, triple := range [][3]float32{
		{math.MaxFloat32, 1, float32(math.Ldexp(1, 103))},
		{-math.MaxFloat32, 1, -float32(math.Ldexp(1, 103))},
		{math.SmallestNonzeroFloat32, .5, 0},
		{-math.SmallestNonzeroFloat32, .5, 0},
		{1, 1, -1},
		{math.MaxFloat32, math.SmallestNonzeroFloat32, -float32(math.Ldexp(1, -21))},
	} {
		a, b, c := triple[0], triple[1], triple[2]
		if math.Float32bits(sincNetFMA32(a, b, c)) != math.Float32bits(oracle(a, b, c)) {
			t.Fatal("FMA boundary", triple)
		}
	}
	t.Logf("exact big.Float comparison: %d finite triples", checked)
}

func TestSincNetNormalizationPinnedTorch(t *testing.T) {
	data, err := os.ReadFile("testdata/sincnet-norm-reference.json.gz")
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
		t.Fatal("invalid norm fixture", err)
	}
	var f struct {
		Schema    int
		Reference struct {
			Commit string `json:"torch_commit"`
			SHA    string `json:"sincnet_sha256"`
		}
		Cases []struct {
			Stride, Stage, Channels, Frames int
			Kind                            string
			Input, Output                   []float32
			Norm                            SincNetNorm
		}
	}
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	if f.Schema != 1 || len(f.Cases) != 28 || f.Reference.Commit != "08187d9e0fba026dc8217405802ab5381dc88d90" || f.Reference.SHA != "3f151a2482c3f8c266b1efd9bdcab297238c52b4792a7265261f2da063e5dade" {
		t.Fatal("norm source contract")
	}
	count := 0
	for _, c := range f.Cases {
		if len(c.Input) != c.Channels*c.Frames || len(c.Output) != len(c.Input) {
			t.Fatal("norm fixture shape")
		}
		x := append([]float32(nil), c.Input...)
		if err := sincNetNorm(context.Background(), x, c.Channels, c.Frames, c.Norm); err != nil {
			t.Fatal(err)
		}
		for i, v := range x {
			if math.Float32bits(v) != math.Float32bits(c.Output[i]) {
				t.Fatalf("norm %s stride%d stage%d index%d got%.9g want%.9g", c.Kind, c.Stride, c.Stage, i, v, c.Output[i])
			}
		}
		count += len(x)
	}
	t.Logf("28 isolated normalization cases: %d bit-exact outputs", count)
	// Exercise all checkpoints of a nonconstant multichannel norm invocation.
	c := f.Cases[1]
	counter := newPowersetContext(0)
	err = sincNetNorm(counter, append([]float32(nil), c.Input...), c.Channels, c.Frames, c.Norm)
	counter.cancel()
	if err != nil {
		t.Fatal(err)
	}
	for at := 1; at <= counter.calls; at++ {
		ctx := newPowersetContext(at)
		err := sincNetNorm(ctx, append([]float32(nil), c.Input...), c.Channels, c.Frames, c.Norm)
		ctx.cancel()
		if !errors.Is(err, context.Canceled) {
			t.Fatal("norm cancellation", at, err)
		}
	}
	t.Logf("normalization cancellation checkpoints: %d", counter.calls)
}
