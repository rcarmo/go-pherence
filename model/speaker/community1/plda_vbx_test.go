package community1

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"sync"
	"testing"
)

type pldaOracle struct {
	Config             PLDAConfig
	Rows               int
	Weights            PreparedPLDAWeights
	Input, Output, Phi []float64
}
type vbxState struct {
	Gamma, Priors, Alpha []float64
	InvL                 []float64 `json:"inv_l"`
	ELBO                 float64
}
type vbxOracle struct {
	Name       string
	Config     VBxConfig
	Input, Phi []float64
	Labels     []int
	Speakers   int
	Iterations []vbxState
}

func loadPLDAVBxOracles(t *testing.T) ([]pldaOracle, []vbxOracle) {
	t.Helper()
	data, err := os.ReadFile("testdata/plda-vbx-reference.json")
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
		PLDA []pldaOracle
		VBx  []vbxOracle
	}
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	if f.Schema != 1 || f.Abs != 2e-10 || f.Rel != 2e-10 || f.Reference.SHA != "a8c644feea4b381f9c1e7da72e0e47775c1fd482067e686801ddc16e5cac3c0e" || len(f.PLDA) != 4 || len(f.VBx) != 10 {
		t.Fatal("PLDA/VBx fixture contract")
	}
	return f.PLDA, f.VBx
}
func closeClustering64(t *testing.T, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatal("clustering length", len(got), len(want))
	}
	for i, v := range got {
		if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v-want[i]) > 2e-10+2e-10*math.Abs(want[i]) {
			t.Fatalf("clustering[%d] got%.17g want%.17g delta%.5g", i, v, want[i], v-want[i])
		}
	}
}
func TestPreparedPLDAPinnedOracle(t *testing.T) {
	cases, _ := loadPLDAVBxOracles(t)
	for _, c := range cases {
		model, err := NewPreparedPLDA(context.Background(), c.Config, c.Weights)
		if err != nil {
			t.Fatal(err)
		}
		inputBefore := append([]float64(nil), c.Input...)
		out, err := model.Transform(context.Background(), c.Input, c.Rows)
		if err != nil {
			t.Fatal(err)
		}
		closeClustering64(t, out, c.Output)
		closeClustering64(t, model.Phi(), c.Phi)
		if !reflect.DeepEqual(inputBefore, c.Input) {
			t.Fatal("changed PLDA input")
		}
		for _, values := range [][]float64{c.Weights.Mean1, c.Weights.Mean2, c.Weights.LDA, c.Weights.Mu, c.Weights.Transform, c.Weights.Phi} {
			for i := range values {
				values[i] = 99
			}
		}
		phi := model.Phi()
		phi[0] = 999
		closeClustering64(t, model.Phi(), c.Phi)
		for i := range out {
			out[i] = 99
		}
		out, err = model.Transform(context.Background(), c.Input, c.Rows)
		if err != nil {
			t.Fatal(err)
		}
		closeClustering64(t, out, c.Output)
	}
}
func TestVBxPinnedIterations(t *testing.T) {
	_, cases := loadPLDAVBxOracles(t)
	boundaries := 0
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			before := append([]float64(nil), c.Input...)
			seen := 0
			result, err := ClusterVBxObserved(context.Background(), c.Input, c.Phi, c.Labels, c.Config, func(i int, q, pi, alpha, inv []float64, elbo float64) error {
				if i != seen || i >= len(c.Iterations) {
					t.Fatal("iteration mismatch")
				}
				seen++
				boundaries++
				want := c.Iterations[i]
				closeClustering64(t, q, want.Gamma)
				closeClustering64(t, pi, want.Priors)
				closeClustering64(t, alpha, want.Alpha)
				closeClustering64(t, inv, want.InvL)
				closeClustering64(t, []float64{elbo}, []float64{want.ELBO})
				for row := 0; row < c.Config.Rows; row++ {
					sum := float64(0)
					for _, v := range q[row*c.Speakers : (row+1)*c.Speakers] {
						if v < 0 || v > 1+1e-12 {
							t.Fatal("invalid responsibility")
						}
						sum += v
					}
					closeClustering64(t, []float64{sum}, []float64{1})
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			if seen != len(c.Iterations) || result.Speakers != c.Speakers || !reflect.DeepEqual(c.Input, before) {
				t.Fatal("VBx state/ownership")
			}
			last := c.Iterations[len(c.Iterations)-1]
			closeClustering64(t, result.Responsibilities, last.Gamma)
			closeClustering64(t, result.Priors, last.Priors)
			stopped, decreased := false, false
			if len(c.Iterations) > 1 {
				for i := 1; i < len(c.Iterations); i++ {
					if c.Iterations[i].ELBO-c.Iterations[i-1].ELBO < 0 {
						decreased = true
					}
				}
				stopped = last.ELBO-c.Iterations[len(c.Iterations)-2].ELBO < c.Config.Epsilon
			}
			if result.StoppedByTolerance != stopped || result.DecreasedELBO != decreased {
				t.Fatal("lost stop diagnostic")
			}
			for _, v := range [][]float64{result.Responsibilities, result.Priors, result.Alpha, result.InvL, result.ELBO} {
				for i := range v {
					v[i] = 99
				}
			}
			fresh, err := ClusterVBx(context.Background(), c.Input, c.Phi, c.Labels, c.Config)
			if err != nil {
				t.Fatal(err)
			}
			closeClustering64(t, fresh.Responsibilities, last.Gamma)
		})
	}
	if boundaries != 45 {
		t.Fatal("changed iteration fixture count", boundaries)
	}
}
func TestPreparedPLDAToVBxBridge(t *testing.T) {
	ps, vs := loadPLDAVBxOracles(t)
	p, v := ps[1], vs[len(vs)-1]
	if v.Name != "prepared_bridge" {
		t.Fatal("missing prepared bridge")
	}
	model, err := NewPreparedPLDA(context.Background(), p.Config, p.Weights)
	if err != nil {
		t.Fatal(err)
	}
	x, err := model.Transform(context.Background(), p.Input, p.Rows)
	if err != nil {
		t.Fatal(err)
	}
	result, err := ClusterVBx(context.Background(), x, model.Phi(), v.Labels, v.Config)
	if err != nil {
		t.Fatal(err)
	}
	want := v.Iterations[len(v.Iterations)-1]
	closeClustering64(t, result.Responsibilities, want.Gamma)
	closeClustering64(t, result.Priors, want.Priors)
	closeClustering64(t, result.ELBO, []float64{v.Iterations[0].ELBO, v.Iterations[1].ELBO, v.Iterations[2].ELBO})
}

func TestPreparedPLDAValidation(t *testing.T) {
	cases, _ := loadPLDAVBxOracles(t)
	if (*PreparedPLDA)(nil).Phi() != nil || new(PreparedPLDA).Phi() != nil {
		t.Fatal("nil/zero Phi")
	}
	base := cases[1]
	for _, kind := range []string{"input_dim", "projected_dim", "output_dim", "mean1", "mean2", "lda", "mu", "transform", "phi", "nan", "inf", "negative_phi", "unordered_phi"} {
		cases, _ := loadPLDAVBxOracles(t)
		c, w := cases[1].Config, cases[1].Weights
		switch kind {
		case "input_dim":
			c.InputDim = int(^uint(0) >> 1)
		case "projected_dim":
			c.ProjectedDim = 0
		case "output_dim":
			c.OutputDim = c.ProjectedDim + 1
		case "mean1":
			w.Mean1 = w.Mean1[1:]
		case "mean2":
			w.Mean2 = nil
		case "lda":
			w.LDA = w.LDA[1:]
		case "mu":
			w.Mu = nil
		case "transform":
			w.Transform = nil
		case "phi":
			w.Phi = nil
		case "nan":
			w.Mean1[0] = math.NaN()
		case "inf":
			w.Transform[0] = math.Inf(1)
		case "negative_phi":
			w.Phi[0] = -1
		case "unordered_phi":
			w.Phi[1] = w.Phi[0] + 1
		}
		if out, err := NewPreparedPLDA(context.Background(), c, w); err == nil || out != nil {
			t.Fatal("accepted prepared coefficients", kind)
		}
	}
	m, err := NewPreparedPLDA(context.Background(), base.Config, base.Weights)
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"rows_zero", "rows_huge", "short", "nan", "infinite", "zero_norm", "overflow"} {
		input := append([]float64(nil), base.Input...)
		rows := base.Rows
		switch kind {
		case "rows_zero":
			rows = 0
		case "rows_huge":
			rows = int(^uint(0) >> 1)
		case "short":
			input = input[1:]
		case "nan":
			input[0] = math.NaN()
		case "infinite":
			input[0] = math.Inf(-1)
		case "zero_norm":
			copy(input, base.Weights.Mean1)
		case "overflow":
			input[0] = math.MaxFloat64
		}
		if out, err := m.Transform(context.Background(), input, rows); err == nil || out != nil {
			t.Fatal("accepted input", kind)
		}
	}
	zero, _ := NewPreparedPLDA(context.Background(), PLDAConfig{1, 1, 1}, PreparedPLDAWeights{[]float64{0}, []float64{0}, []float64{0}, []float64{0}, []float64{1}, []float64{1}})
	if out, err := zero.Transform(context.Background(), []float64{1}, 1); err == nil || out != nil {
		t.Fatal("accepted second zero norm")
	}
	if out, err := (*PreparedPLDA)(nil).Transform(context.Background(), nil, 1); err == nil || out != nil {
		t.Fatal("nil receiver")
	}
	if out, err := new(PreparedPLDA).Transform(context.Background(), nil, 1); err == nil || out != nil {
		t.Fatal("zero receiver")
	}
}
func TestVBxValidation(t *testing.T) {
	for _, kind := range []string{"rows", "dimension", "iterations0", "iterations101", "length", "phi_length", "labels_length", "negative_label", "large_label", "negative_phi", "nan_x", "inf_phi", "Fa_zero", "Fb_negative", "ratio_overflow", "ratio_underflow", "epsilon_negative", "smooth_nan", "epsilon_inf", "x_overflow"} {
		_, cases := loadPLDAVBxOracles(t)
		c := cases[0]
		cfg := c.Config
		switch kind {
		case "rows":
			cfg.Rows = int(^uint(0) >> 1)
		case "dimension":
			cfg.Dimension = 513
		case "iterations0":
			cfg.MaxIterations = 0
		case "iterations101":
			cfg.MaxIterations = 101
		case "length":
			c.Input = c.Input[1:]
		case "phi_length":
			c.Phi = c.Phi[1:]
		case "labels_length":
			c.Labels = c.Labels[1:]
		case "negative_label":
			c.Labels[0] = -1
		case "large_label":
			c.Labels[0] = 64
		case "negative_phi":
			c.Phi[0] = -1
		case "nan_x":
			c.Input[0] = math.NaN()
		case "inf_phi":
			c.Phi[0] = math.Inf(-1)
		case "Fa_zero":
			cfg.Fa = 0
		case "Fb_negative":
			cfg.Fb = -1
		case "ratio_overflow":
			cfg.Fa = math.MaxFloat64
			cfg.Fb = math.SmallestNonzeroFloat64
		case "ratio_underflow":
			cfg.Fa = math.SmallestNonzeroFloat64
			cfg.Fb = math.MaxFloat64
		case "epsilon_negative":
			cfg.Epsilon = -1
		case "smooth_nan":
			cfg.InitSmoothing = math.NaN()
		case "epsilon_inf":
			cfg.Epsilon = math.Inf(1)
		case "x_overflow":
			c.Input[0] = math.MaxFloat64
		}
		if out, err := ClusterVBx(context.Background(), c.Input, c.Phi, c.Labels, cfg); err == nil || out != nil {
			t.Fatal("accepted VBx", kind)
		}
	}
	cfg := Community1VBxConfig(256, 512)
	cfg.MaxIterations = 100
	labels := make([]int, 256)
	labels[0] = 63
	if out, err := ClusterVBx(context.Background(), make([]float64, 256*512), make([]float64, 512), labels, cfg); err == nil || out != nil {
		t.Fatal("work bound not enforced")
	}
	// Maximum label, gaps and representable enormous smoothing remain stable.
	cfg = Community1VBxConfig(1, 1)
	cfg.InitSmoothing = math.MaxFloat64
	out, err := ClusterVBx(context.Background(), []float64{.2}, []float64{0}, []int{63}, cfg)
	if err != nil || out.Speakers != 64 {
		t.Fatal("max label/smoothing", err)
	}
}
func TestPLDAVBxCancellationAndOwnership(t *testing.T) {
	ps, vs := loadPLDAVBxOracles(t)
	p, v := ps[0], vs[0]
	model, err := NewPreparedPLDA(context.Background(), p.Config, p.Weights)
	if err != nil {
		t.Fatal(err)
	}
	calls := []func(context.Context) error{
		func(ctx context.Context) error {
			out, err := NewPreparedPLDA(ctx, p.Config, p.Weights)
			if err != nil && out != nil {
				t.Fatal("partial constructor")
			}
			return err
		},
		func(ctx context.Context) error {
			out, err := model.Transform(ctx, p.Input, p.Rows)
			if err != nil && out != nil {
				t.Fatal("partial transform")
			}
			return err
		},
		func(ctx context.Context) error {
			out, err := ClusterVBx(ctx, v.Input, v.Phi, v.Labels, v.Config)
			if err != nil && out != nil {
				t.Fatal("partial VBx")
			}
			return err
		},
	}
	for stage, run := range calls {
		count := newPowersetContext(0)
		err := run(count)
		count.cancel()
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("PLDA/VBx stage%d checkpoints%d", stage, count.calls)
		for at := 1; at <= count.calls; at++ {
			ctx := newPowersetContext(at)
			err := run(ctx)
			ctx.cancel()
			if !errors.Is(err, context.Canceled) {
				t.Fatal("cancellation", stage, at, err)
			}
		}
	}
	sentinel := errors.New("observer rejected")
	if out, err := ClusterVBxObserved(context.Background(), v.Input, v.Phi, v.Labels, v.Config, func(int, []float64, []float64, []float64, []float64, float64) error { return sentinel }); out != nil || !errors.Is(err, sentinel) {
		t.Fatal("observer cause", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if out, err := ClusterVBxObserved(ctx, v.Input, v.Phi, v.Labels, v.Config, func(int, []float64, []float64, []float64, []float64, float64) error { cancel(); return nil }); out != nil || !errors.Is(err, context.Canceled) {
		t.Fatal("observer cancellation", err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out, err := model.Transform(context.Background(), p.Input, p.Rows)
			if err != nil {
				t.Error(err)
				return
			}
			closeClustering64(t, out, p.Output)
			result, err := ClusterVBx(context.Background(), v.Input, v.Phi, v.Labels, v.Config)
			if err != nil {
				t.Error(err)
				return
			}
			closeClustering64(t, result.Responsibilities, v.Iterations[len(v.Iterations)-1].Gamma)
		}()
	}
	wg.Wait()
}
