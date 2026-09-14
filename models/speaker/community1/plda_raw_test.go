package community1

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"testing"
)

type rawPLDAOracle struct {
	Config                                  PLDAConfig
	Rows                                    int
	Raw                                     RawPLDAWeights
	Input, Direct, Gram, Phi, Gamma, Priors []float64
	Order                                   []int
	Condition                               float64
}

func loadRawPLDAOracles(t *testing.T) []rawPLDAOracle {
	t.Helper()
	data, err := os.ReadFile("testdata/raw-plda-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Schema    int
		Abs       float64 `json:"absolute_tolerance"`
		Rel       float64 `json:"relative_tolerance"`
		Reference struct {
			SHA string `json:"vbx_sha256"`
		}
		Cases []rawPLDAOracle
	}
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	if f.Schema != 1 || f.Abs != 2e-10 || f.Rel != 2e-10 || f.Reference.SHA != "a8c644feea4b381f9c1e7da72e0e47775c1fd482067e686801ddc16e5cac3c0e" || len(f.Cases) != 6 {
		t.Fatal("raw PLDA fixture contract")
	}
	return f.Cases
}
func TestRawPLDAPinnedBasisInvariantOracle(t *testing.T) {
	for _, c := range loadRawPLDAOracles(t) {
		result, err := PrepareRawPLDA(context.Background(), c.Config, c.Raw)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(result.Order, c.Order) || result.InverseResidual > 1e-8 {
			t.Fatal("raw preparation metadata")
		}
		closeClustering64(t, []float64{result.ConditionInf}, []float64{c.Condition})
		closeClustering64(t, result.Model.Phi(), c.Phi)
		x, err := result.Model.Transform(context.Background(), c.Input, c.Rows)
		if err != nil {
			t.Fatal(err)
		}
		closeClustering64(t, x, c.Direct)
		gram := make([]float64, c.Rows*c.Rows)
		for i := 0; i < c.Rows; i++ {
			for j := 0; j < c.Rows; j++ {
				for d := 0; d < c.Config.OutputDim; d++ {
					gram[i*c.Rows+j] += x[i*c.Config.OutputDim+d] * x[j*c.Config.OutputDim+d]
				}
			}
		}
		closeClustering64(t, gram, c.Gram)
		vb, err := ClusterVBx(context.Background(), x, result.Model.Phi(), []int{0, 0, 0, 1, 1, 1, 1}, Community1VBxConfig(c.Rows, c.Config.OutputDim))
		if err != nil {
			t.Fatal(err)
		}
		closeClustering64(t, vb.Responsibilities, c.Gamma)
		closeClustering64(t, vb.Priors, c.Priors)
		for _, values := range [][]float64{c.Raw.Mean1, c.Raw.Mean2, c.Raw.LDA, c.Raw.Mu, c.Raw.TR, c.Raw.Psi} {
			for i := range values {
				values[i] = 99
			}
		}
		after, err := result.Model.Transform(context.Background(), c.Input, c.Rows)
		if err != nil {
			t.Fatal(err)
		}
		closeClustering64(t, after, c.Direct)
	}
}
func TestRawPLDANPZNumericLifetime(t *testing.T) {
	c := loadRawPLDAOracles(t)[0]
	for _, tag := range []string{"lef8", "bef8", "lef4", "bef4"} {
		xb, err := os.ReadFile("testdata/raw-plda-npz/" + tag + "-xvec.npz")
		if err != nil {
			t.Fatal(err)
		}
		pb, err := os.ReadFile("testdata/raw-plda-npz/" + tag + "-plda.npz")
		if err != nil {
			t.Fatal(err)
		}
		result, err := LoadRawPLDANPZ(context.Background(), bytes.NewReader(xb), int64(len(xb)), bytes.NewReader(pb), int64(len(pb)), c.Config)
		if err != nil {
			t.Fatal(tag, err)
		}
		for i := range xb {
			xb[i] = 0
		}
		for i := range pb {
			pb[i] = 0
		}
		out, err := result.Model.Transform(context.Background(), c.Input, c.Rows)
		if err != nil {
			t.Fatal(err)
		}
		if tag == "lef8" || tag == "bef8" {
			closeClustering64(t, out, c.Direct)
		} else {
			// Test exact widening/ownership, not mixed-dtype reference inference parity.
			raw := loadRawPLDAOracles(t)[0].Raw
			for _, values := range [][]float64{raw.Mean1, raw.Mean2, raw.LDA, raw.Mu, raw.TR, raw.Psi} {
				for i, v := range values {
					values[i] = float64(float32(v))
				}
			}
			expected, err := PrepareRawPLDA(context.Background(), c.Config, raw)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(result.Model, expected.Model) {
				t.Fatal("F32 widening mismatch")
			}
		}
	}
}
func TestRawPLDAValidation(t *testing.T) {
	for _, kind := range []string{"config", "length", "nan", "psi_zero", "psi_negative", "singular", "illconditioned", "degenerate_cut", "close_cut", "infinite"} {
		c := loadRawPLDAOracles(t)[0]
		n := c.Config.ProjectedDim
		switch kind {
		case "config":
			c.Config.InputDim = int(^uint(0) >> 1)
		case "length":
			c.Raw.TR = c.Raw.TR[1:]
		case "nan":
			c.Raw.Mean1[0] = math.NaN()
		case "psi_zero":
			c.Raw.Psi[0] = 0
		case "psi_negative":
			c.Raw.Psi[0] = -1
		case "singular":
			copy(c.Raw.TR[n:2*n], c.Raw.TR[:n])
		case "illconditioned":
			clear(c.Raw.TR)
			for i := 0; i < n; i++ {
				c.Raw.TR[i*n+i] = 1
			}
			c.Raw.TR[0] = 1e-8
		case "degenerate_cut":
			for i := range c.Raw.Psi {
				c.Raw.Psi[i] = 1
			}
		case "close_cut":
			c.Raw.Psi = []float64{2, 1, 1 + 1e-10}
		case "infinite":
			c.Raw.TR[0] = math.Inf(1)
		}
		if out, err := PrepareRawPLDA(context.Background(), c.Config, c.Raw); err == nil || out != nil {
			t.Fatal("accepted rawPLDA", kind)
		}
	}
	// Uniform scale must not be mistaken for poor conditioning; pivoting required.
	for _, scale := range []float64{1e-50, 1, 1e50} {
		condition, residual, err := rawPLDACondition(context.Background(), []float64{0, scale, scale, 0}, 2)
		if err != nil || condition != 1 || residual > 1e-8 {
			t.Fatal("scaled pivot", condition, residual, err)
		}
	}
	if out, err := LoadRawPLDANPZ(context.Background(), bytes.NewReader([]byte("bad")), 3, nil, 0, PLDAConfig{1, 1, 1}); err == nil || out != nil {
		t.Fatal("bad container")
	}
}
func TestRawPLDACancellation(t *testing.T) {
	c := loadRawPLDAOracles(t)[0]
	xb, _ := os.ReadFile("testdata/raw-plda-npz/lef8-xvec.npz")
	pb, _ := os.ReadFile("testdata/raw-plda-npz/lef8-plda.npz")
	runs := []func(context.Context) error{
		func(ctx context.Context) error {
			out, err := PrepareRawPLDA(ctx, c.Config, c.Raw)
			if err != nil && out != nil {
				t.Fatal("partial raw model")
			}
			return err
		},
		func(ctx context.Context) error {
			out, err := LoadRawPLDANPZ(ctx, bytes.NewReader(xb), int64(len(xb)), bytes.NewReader(pb), int64(len(pb)), c.Config)
			if err != nil && out != nil {
				t.Fatal("partial npz model")
			}
			return err
		},
	}
	for stage, run := range runs {
		counter := newPowersetContext(0)
		err := run(counter)
		counter.cancel()
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("raw PLDA stage%d checkpoints%d", stage, counter.calls)
		for at := 1; at <= counter.calls; at++ {
			ctx := newPowersetContext(at)
			err := run(ctx)
			ctx.cancel()
			if !errors.Is(err, context.Canceled) {
				t.Fatal("cancel", stage, at, err)
			}
		}
	}
}
