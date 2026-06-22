package minicpmv

import "os"

const MiniCPMOFixturePath = "model/minicpmv/testdata/minicpmo_fixture"

func LoadMiniCPMOFixtureMetadata() (Metadata, error) {
	if _, err := os.Stat(MiniCPMOFixturePath); err == nil {
		return LoadMetadata(MiniCPMOFixturePath)
	}
	return LoadMetadata("testdata/minicpmo_fixture")
}
