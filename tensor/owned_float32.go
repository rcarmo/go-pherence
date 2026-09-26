package tensor

// FromOwnedFloat32 creates a tensor from a float32 slice, taking exclusive
// ownership of the backing storage without copying the payload. The caller must
// not read, mutate, or reuse data after calling this function.
func FromOwnedFloat32(data []float32, shape []int) *Tensor {
	n := shapeSize(shape)
	if n < 0 || n != len(data) {
		panic("shape mismatch")
	}
	u := BufferOp(Float32, shape)
	u.buf = &Buffer{
		Data:   float32ToByteSlice(data),
		DType:  Float32,
		Length: len(data),
	}
	return &Tensor{uop: u, shape: NewShape(shape)}
}
