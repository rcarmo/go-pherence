package arm64

func init() {
	HasVecAsm = RuntimeCapabilities().HasVec
}
