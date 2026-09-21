package qwenimage21

import "fmt"

type ImageShape struct{ Height, Width int }
type Segment struct{ Start, End, ContextStart, ImageIndex int }
type Position struct{ Frame, Height, Width int }
type Layout struct {
	Segments     []Segment
	Positions    []Position
	PrefixLength int
}

// BuildLayout mirrors Qwen Image 2.1's block-causal text/reference/target layout.
// ImageSlots uses 0 for text and one-based contiguous reference-image tags.
// The last ImageShape is always the target latent.
func BuildLayout(textLength int, imageSlots []int, imageShapes []ImageShape) (Layout, error) {
	if textLength < 0 || len(imageShapes) == 0 || (len(imageSlots) > 0 && len(imageSlots) != textLength) {
		return Layout{}, fmt.Errorf("qwen-image-2.1: invalid image token layout")
	}
	at := func(i int) int {
		if len(imageSlots) == 0 {
			return 0
		}
		return imageSlots[i]
	}
	var out Layout
	position, nextImage := 0, 0
	appendImage := func(index, contextStart int) {
		shape := imageShapes[index]
		start := len(out.Positions)
		out.Segments = append(out.Segments, Segment{start, start + shape.Height*shape.Width, contextStart, index})
		for h := 0; h < shape.Height; h++ {
			for w := 0; w < shape.Width; w++ {
				out.Positions = append(out.Positions, Position{position, h - (shape.Height - shape.Height/2), w - (shape.Width - shape.Width/2)})
			}
		}
		position += max(shape.Height, shape.Width)
	}
	for i := 0; i < textLength; {
		tag := at(i)
		begin := i
		i++
		for i < textLength && at(i) == tag {
			i++
		}
		if tag != 0 {
			if tag != nextImage+1 || nextImage+1 >= len(imageShapes) || (i-begin)*4 != imageShapes[nextImage].Height*imageShapes[nextImage].Width {
				return Layout{}, fmt.Errorf("qwen-image-2.1: vision slots and reference latents must match")
			}
			appendImage(nextImage, begin)
			nextImage++
		} else {
			start := len(out.Positions)
			out.Segments = append(out.Segments, Segment{start, start + i - begin, begin, -1})
			for j := begin; j < i; j++ {
				out.Positions = append(out.Positions, Position{position, position, position})
				position++
			}
		}
	}
	if nextImage+1 != len(imageShapes) {
		return Layout{}, fmt.Errorf("qwen-image-2.1: missing reference image slots")
	}
	out.PrefixLength = len(out.Positions)
	appendImage(nextImage, textLength)
	return out, nil
}

// AttentionAllowed is the exact block-causal relation: causal globally, but
// each image block is internally bidirectional.
func (l Layout) AttentionAllowed(query, key int) bool {
	if query < 0 || key < 0 || query >= len(l.Positions) || key >= len(l.Positions) {
		return false
	}
	if query >= key {
		return true
	}
	for _, s := range l.Segments {
		if s.ImageIndex >= 0 && query >= s.Start && query < s.End && key >= s.Start && key < s.End {
			return true
		}
	}
	return false
}

// TokenMetadata labels text as -1 and each image block by its image index.
func (l Layout) TokenMetadata() ([]int, []bool) {
	ids := make([]int, len(l.Positions))
	for i := range ids {
		ids[i] = -1
	}
	target := make([]bool, len(ids))
	lastImage := -1
	for _, s := range l.Segments {
		if s.ImageIndex < 0 {
			continue
		}
		lastImage = max(lastImage, s.ImageIndex)
		for i := s.Start; i < s.End; i++ {
			ids[i] = s.ImageIndex
		}
	}
	for _, s := range l.Segments {
		if s.ImageIndex == lastImage {
			for i := s.Start; i < s.End; i++ {
				target[i] = true
			}
		}
	}
	return ids, target
}
