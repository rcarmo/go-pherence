//go:build amd64

package omnivoice

// A multiple of the six-row packed kernel avoids two NN fallback rows per tile.
const codecPackedTransposeTile = 30
