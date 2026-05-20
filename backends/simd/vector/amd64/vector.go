package amd64

func init() {
	HasVecAsm = RuntimeCapabilities().HasVec
}
