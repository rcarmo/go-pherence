package ideogram4

import "fmt"

func (c Config) NewLatents(batch, height, width int) (Latents, error) {
	if batch <= 0 {
		return Latents{}, fmt.Errorf("invalid Ideogram4 batch size %d", batch)
	}
	tokens, err := c.LatentTokenCount(height, width)
	if err != nil {
		return Latents{}, err
	}
	return Latents{Batch: batch, Tokens: tokens, Channels: c.InChannels, Data: make([]float32, batch*tokens*c.InChannels)}, nil
}

func (c Config) ValidateLatents(z Latents) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if z.Batch <= 0 || z.Tokens < 0 || z.Channels != c.InChannels {
		return fmt.Errorf("invalid Ideogram4 latents shape: batch=%d tokens=%d channels=%d want_channels=%d", z.Batch, z.Tokens, z.Channels, c.InChannels)
	}
	want := z.Batch * z.Tokens * z.Channels
	if want < 0 || len(z.Data) != want {
		return fmt.Errorf("invalid Ideogram4 latents len=%d want=%d", len(z.Data), want)
	}
	return nil
}
