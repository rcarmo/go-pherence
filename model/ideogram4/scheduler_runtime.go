package ideogram4

import "fmt"

// FlowMatchScheduler is a concrete, weight-free implementation of the Scheduler
// interface. It derives FlowMatch timesteps from the Ideogram4 logit-normal
// schedule and performs the Euler latent update used by the sampling loop.
//
// Unlike the DiT/VAE paths, the scheduler needs no model weights, so it is
// implemented natively here rather than behind a backend boundary.
type FlowMatchScheduler struct {
	Height   int
	Width    int
	Config   Config
	Schedule LogitNormalSchedule
}

// NewFlowMatchScheduler builds a scheduler for the given target resolution,
// resolving the default logit-normal schedule when one is not supplied.
func NewFlowMatchScheduler(cfg Config, height, width int, schedule LogitNormalSchedule) (*FlowMatchScheduler, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if height <= 0 || width <= 0 {
		return nil, fmt.Errorf("invalid Ideogram4 scheduler resolution %dx%d", height, width)
	}
	if schedule.Std == 0 {
		var err error
		schedule, err = DefaultScheduleForResolution(height, width, 512, 512, 1, 1)
		if err != nil {
			return nil, err
		}
	}
	if err := schedule.Validate(); err != nil {
		return nil, err
	}
	return &FlowMatchScheduler{Height: height, Width: width, Config: cfg, Schedule: schedule}, nil
}

// Steps satisfies the Scheduler interface, returning ordered FlowMatch steps
// (high noise first) for the requested step count.
func (s *FlowMatchScheduler) Steps(numSteps int) ([]FlowStep, error) {
	if s == nil {
		return nil, ErrRuntimeNotImplemented
	}
	plan, err := s.Config.BuildSamplingPlan(s.Height, s.Width, numSteps, 1, nil, DefaultMaxTextTokens, s.Schedule)
	if err != nil {
		return nil, err
	}
	return plan.Steps, nil
}

// Step applies one Euler update: x_{t-1} = x_t + sigma * velocity, where sigma
// is the (negative) timestep delta carried by the FlowStep. The latent and
// velocity tensors must share identical shape.
func (s *FlowMatchScheduler) Step(latents Latents, velocity Latents, step FlowStep) (Latents, error) {
	if s == nil {
		return Latents{}, ErrRuntimeNotImplemented
	}
	if err := latents.validate(); err != nil {
		return Latents{}, err
	}
	if err := velocity.validate(); err != nil {
		return Latents{}, err
	}
	if latents.Batch != velocity.Batch || latents.Tokens != velocity.Tokens || latents.Channels != velocity.Channels {
		return Latents{}, fmt.Errorf("Ideogram4 scheduler shape mismatch latent=%dx%dx%d velocity=%dx%dx%d",
			latents.Batch, latents.Tokens, latents.Channels, velocity.Batch, velocity.Tokens, velocity.Channels)
	}
	out := Latents{Batch: latents.Batch, Tokens: latents.Tokens, Channels: latents.Channels, Data: make([]float32, len(latents.Data))}
	sigma := step.Sigma
	for i := range latents.Data {
		out.Data[i] = latents.Data[i] + sigma*velocity.Data[i]
	}
	return out, nil
}

func (l Latents) validate() error {
	if l.Batch <= 0 || l.Tokens <= 0 || l.Channels <= 0 {
		return fmt.Errorf("invalid Ideogram4 latents shape %dx%dx%d", l.Batch, l.Tokens, l.Channels)
	}
	want := l.Batch * l.Tokens * l.Channels
	if len(l.Data) != want {
		return fmt.Errorf("Ideogram4 latents data=%d want=%d", len(l.Data), want)
	}
	return nil
}
