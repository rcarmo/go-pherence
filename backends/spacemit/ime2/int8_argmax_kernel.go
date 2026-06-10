//go:build riscv64

package ime2

// VmadotI8ArgmaxGroups1024 computes nGroups contiguous 8-row INT8 row groups
// and returns the maximum raw int32 dot and row index within this shard.
//
//go:noescape
func VmadotI8ArgmaxGroups1024(wPacked, actPacked *byte, scratch *int32, bestVal *int32, bestID *int64, nGroups, K, rowStart int)
