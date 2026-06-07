package ime2

// VmadotI8Groups1024 computes nGroups contiguous 8-row INT8 row groups with
// one assembly dispatch. wPacked is native PackTiles1024 layout, actPacked is
// the broadcast activation tile layout, scratch is [64]int32, out is float32.
//
//go:noescape
func VmadotI8Groups1024(wPacked, actPacked *byte, scratch *int32, out, scale *float32, nGroups, K int)
