package servingbench

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// ArrivalMode controls how request inter-arrival times are generated.
type ArrivalMode string

const (
	ArrivalFixed   ArrivalMode = "fixed"
	ArrivalPoisson ArrivalMode = "poisson"
	ArrivalGamma   ArrivalMode = "gamma"
)

// ArrivalConfig describes the request arrival process.
type ArrivalConfig struct {
	Mode       ArrivalMode `json:"mode"`
	Rate       float64     `json:"rate"`
	GammaShape float64     `json:"gamma_shape,omitempty"`
}

func (cfg ArrivalConfig) normalized() ArrivalConfig {
	if cfg.Mode == "" {
		cfg.Mode = ArrivalFixed
	}
	if cfg.Mode == ArrivalGamma && cfg.GammaShape == 0 {
		cfg.GammaShape = 2
	}
	return cfg
}

// Validate checks the arrival configuration.
func (cfg ArrivalConfig) Validate() error {
	cfg = cfg.normalized()
	if cfg.Rate <= 0 || math.IsNaN(cfg.Rate) || math.IsInf(cfg.Rate, 0) {
		return fmt.Errorf("arrival rate must be positive")
	}
	switch cfg.Mode {
	case ArrivalFixed, ArrivalPoisson:
		return nil
	case ArrivalGamma:
		if cfg.GammaShape <= 0 || math.IsNaN(cfg.GammaShape) || math.IsInf(cfg.GammaShape, 0) {
			return fmt.Errorf("gamma shape must be positive")
		}
		return nil
	default:
		return fmt.Errorf("unsupported arrival mode %q", cfg.Mode)
	}
}

// GenerateArrivalOffsets returns cumulative request arrival offsets for count
// requests. The first request always arrives at t=0 and subsequent arrivals are
// spaced by the configured inter-arrival distribution.
func GenerateArrivalOffsets(count int, cfg ArrivalConfig, seed int64) ([]time.Duration, error) {
	if count < 0 {
		return nil, fmt.Errorf("count must be non-negative")
	}
	cfg = cfg.normalized()
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if count == 0 {
		return []time.Duration{}, nil
	}

	offsets := make([]time.Duration, count)
	meanSeconds := 1 / cfg.Rate
	rng := rand.New(rand.NewSource(seed))
	var elapsed float64
	for i := 1; i < count; i++ {
		elapsed += nextInterArrivalSeconds(rng, cfg, meanSeconds)
		offsets[i] = time.Duration(elapsed * float64(time.Second))
	}
	return offsets, nil
}

func nextInterArrivalSeconds(rng *rand.Rand, cfg ArrivalConfig, meanSeconds float64) float64 {
	switch cfg.Mode {
	case ArrivalFixed:
		return meanSeconds
	case ArrivalPoisson:
		return rng.ExpFloat64() * meanSeconds
	case ArrivalGamma:
		shape := cfg.GammaShape
		scale := meanSeconds / shape
		return sampleGamma(rng, shape) * scale
	default:
		return meanSeconds
	}
}

// sampleGamma draws from Gamma(shape, 1) using Marsaglia and Tsang's method.
func sampleGamma(rng *rand.Rand, shape float64) float64 {
	if shape <= 0 {
		return 0
	}
	if shape < 1 {
		u := rng.Float64()
		return sampleGamma(rng, shape+1) * math.Pow(u, 1/shape)
	}
	d := shape - 1.0/3.0
	c := 1 / math.Sqrt(9*d)
	for {
		x := rng.NormFloat64()
		v := 1 + c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := rng.Float64()
		if u < 1-0.0331*x*x*x*x {
			return d * v
		}
		if math.Log(u) < 0.5*x*x+d*(1-v+math.Log(v)) {
			return d * v
		}
	}
}
