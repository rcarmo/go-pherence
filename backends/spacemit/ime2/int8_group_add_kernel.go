package ime2

// VmadotI8GroupsAdd1024 computes nGroups contiguous 8-row INT8 row groups and
// adds the scaled float result into the existing output buffer.
//
//go:noescape
func VmadotI8GroupsAdd1024(wPacked, actPacked *byte, scratch *int32, out, scale *float32, nGroups, K int)
