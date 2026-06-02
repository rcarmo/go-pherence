package main

// vmadotI8GroupsI321024 computes nGroups contiguous 8-row INT8 row groups and
// writes the 8 diagonal int32 results per group to out.
//
//go:noescape
func vmadotI8GroupsI321024(wPacked, actPacked *byte, scratch, out *int32, nGroups, K int)
