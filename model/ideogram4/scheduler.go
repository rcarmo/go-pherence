package ideogram4

import (
	"fmt"
	"math"
)

type LogitNormalSchedule struct {
	Mean      float64
	Std       float64
	LogSNRMin float64
	LogSNRMax float64
}

func DefaultScheduleForResolution(height, width int, knownHeight, knownWidth int, knownMean, std float64) (LogitNormalSchedule, error) {
	if height <= 0 || width <= 0 || knownHeight <= 0 || knownWidth <= 0 {
		return LogitNormalSchedule{}, fmt.Errorf("invalid Ideogram4 schedule resolution %dx%d known=%dx%d", height, width, knownHeight, knownWidth)
	}
	if std <= 0 {
		std = 1
	}
	if knownMean == 0 {
		knownMean = 1
	}
	numPixels := float64(height * width)
	knownPixels := float64(knownHeight * knownWidth)
	return LogitNormalSchedule{Mean: knownMean + 0.5*math.Log(numPixels/knownPixels), Std: std, LogSNRMin: -15, LogSNRMax: 18}, nil
}

func (s LogitNormalSchedule) Validate() error {
	if s.Std <= 0 {
		return fmt.Errorf("invalid Ideogram4 schedule std %f", s.Std)
	}
	if s.LogSNRMax <= s.LogSNRMin {
		return fmt.Errorf("invalid Ideogram4 logsnr bounds min=%f max=%f", s.LogSNRMin, s.LogSNRMax)
	}
	return nil
}

func (s LogitNormalSchedule) Value(u float64) (float32, error) {
	if err := s.Validate(); err != nil {
		return 0, err
	}
	if u < 0 || u > 1 {
		return 0, fmt.Errorf("invalid Ideogram4 schedule input %f", u)
	}
	z := normalQuantile(u)
	y := s.Mean + s.Std*z
	t := 1 - sigmoid(y)
	tMin := 1.0 / (1 + math.Exp(0.5*s.LogSNRMax))
	tMax := 1.0 / (1 + math.Exp(0.5*s.LogSNRMin))
	if t < tMin {
		t = tMin
	}
	if t > tMax {
		t = tMax
	}
	return float32(t), nil
}

type SamplingPlan struct {
	Height           int
	Width            int
	Steps            []FlowStep
	GuidanceSchedule []float32
	GridH            int
	GridW            int
	ImageTokens      int
	MaxTextTokens    int
}

func (c Config) BuildSamplingPlan(height, width, numSteps int, guidanceScale float32, guidanceSchedule []float32, maxTextTokens int, schedule LogitNormalSchedule) (SamplingPlan, error) {
	if err := c.Validate(); err != nil {
		return SamplingPlan{}, err
	}
	if numSteps <= 0 {
		return SamplingPlan{}, fmt.Errorf("invalid Ideogram4 step count %d", numSteps)
	}
	if maxTextTokens <= 0 {
		maxTextTokens = DefaultMaxTextTokens
	}
	if len(guidanceSchedule) > 0 && len(guidanceSchedule) != numSteps {
		return SamplingPlan{}, fmt.Errorf("Ideogram4 guidance schedule length=%d want steps=%d", len(guidanceSchedule), numSteps)
	}
	gridH, gridW, err := c.LatentGrid(height, width)
	if err != nil {
		return SamplingPlan{}, err
	}
	if schedule.Std == 0 {
		schedule, err = DefaultScheduleForResolution(height, width, 512, 512, 1, 1)
		if err != nil {
			return SamplingPlan{}, err
		}
	}
	if err := schedule.Validate(); err != nil {
		return SamplingPlan{}, err
	}
	steps := make([]FlowStep, 0, numSteps)
	for i := numSteps - 1; i >= 0; i-- {
		tVal, err := schedule.Value(float64(i+1) / float64(numSteps))
		if err != nil {
			return SamplingPlan{}, err
		}
		sVal, err := schedule.Value(float64(i) / float64(numSteps))
		if err != nil {
			return SamplingPlan{}, err
		}
		steps = append(steps, FlowStep{Index: i, Sigma: sVal - tVal, T: tVal})
	}
	gw := make([]float32, numSteps)
	if len(guidanceSchedule) > 0 {
		copy(gw, guidanceSchedule)
	} else {
		for i := range gw {
			gw[i] = guidanceScale
		}
	}
	return SamplingPlan{Height: height, Width: width, Steps: steps, GuidanceSchedule: gw, GridH: gridH, GridW: gridW, ImageTokens: gridH * gridW, MaxTextTokens: maxTextTokens}, nil
}

type CFGLayout struct {
	PositiveTokens int
	NegativeTokens int
	TextTokens     int
	ImageTokens    int
	Asymmetric     bool
}

func (p SamplingPlan) CFGLayout(textTokens int) (CFGLayout, error) {
	if textTokens < 0 || textTokens > p.MaxTextTokens {
		return CFGLayout{}, fmt.Errorf("invalid Ideogram4 text tokens=%d max=%d", textTokens, p.MaxTextTokens)
	}
	if p.ImageTokens <= 0 {
		return CFGLayout{}, fmt.Errorf("invalid Ideogram4 image tokens %d", p.ImageTokens)
	}
	return CFGLayout{PositiveTokens: p.MaxTextTokens + p.ImageTokens, NegativeTokens: p.ImageTokens, TextTokens: textTokens, ImageTokens: p.ImageTokens, Asymmetric: true}, nil
}

func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

// normalQuantile is Peter J. Acklam's rational approximation for the inverse
// standard normal CDF. It is sufficient for deterministic schedule scaffolding
// and avoids pulling in a heavy statistics dependency.
func normalQuantile(p float64) float64 {
	if p <= 0 {
		return math.Inf(-1)
	}
	if p >= 1 {
		return math.Inf(1)
	}
	const a1 = -3.969683028665376e+01
	const a2 = 2.209460984245205e+02
	const a3 = -2.759285104469687e+02
	const a4 = 1.383577518672690e+02
	const a5 = -3.066479806614716e+01
	const a6 = 2.506628277459239e+00
	const b1 = -5.447609879822406e+01
	const b2 = 1.615858368580409e+02
	const b3 = -1.556989798598866e+02
	const b4 = 6.680131188771972e+01
	const b5 = -1.328068155288572e+01
	const c1 = -7.784894002430293e-03
	const c2 = -3.223964580411365e-01
	const c3 = -2.400758277161838e+00
	const c4 = -2.549732539343734e+00
	const c5 = 4.374664141464968e+00
	const c6 = 2.938163982698783e+00
	const d1 = 7.784695709041462e-03
	const d2 = 3.224671290700398e-01
	const d3 = 2.445134137142996e+00
	const d4 = 3.754408661907416e+00
	const plow = 0.02425
	const phigh = 1 - plow
	if p < plow {
		q := math.Sqrt(-2 * math.Log(p))
		return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q + c6) / ((((d1*q+d2)*q+d3)*q+d4)*q + 1)
	}
	if p > phigh {
		q := math.Sqrt(-2 * math.Log(1-p))
		return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q + c6) / ((((d1*q+d2)*q+d3)*q+d4)*q + 1)
	}
	q := p - 0.5
	r := q * q
	return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r + a6) * q / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r + 1)
}
