//go:build !amd64

package omnivoice

func snakePost(row, sine []float32, scale float32) { snakePostVectors(row, sine, scale) }
