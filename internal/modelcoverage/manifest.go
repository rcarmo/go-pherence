package modelcoverage

// Manifest is the JSON schema for docs/model-coverage-manifest.json.
type Manifest struct {
	Version  int                       `json:"version"`
	Families map[string]ManifestFamily `json:"families"`
}

// ManifestFamily captures one model family's coverage and validation metadata.
type ManifestFamily struct {
	Status            string          `json:"status"`
	RuntimeGeneration bool            `json:"runtime_generation"`
	ValidationTarget  string          `json:"validation_target"`
	Packages          []string        `json:"packages"`
	Coverage          map[string]bool `json:"coverage"`
	Commands          []string        `json:"commands"`
}
