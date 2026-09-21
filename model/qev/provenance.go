package qev

// Provenance is the immutable source and artifact contract for QEV v1.
type Provenance struct {
	ModelRepository       string
	ModelRevision         string
	ModelFile             string
	ModelSHA256           string
	ModelBytes            int64
	ModelLicense          string
	TokenizerRepository   string
	TokenizerRevision     string
	TokenizerSHA256       string
	TokenizerConfigSHA256 string
	ChatTemplateSHA256    string
	LlamaRepository       string
	LlamaRevision         string
	LlamaLicense          string
	RLCDRepository        string
	RLCDRevision          string
	RLCDLicense           string
	PlaygroundRepository  string
	PlaygroundRevision    string
	PlaygroundLicense     string
}

var V1Provenance = Provenance{
	ModelRepository:       "unsloth/gemma-4-12b-it-GGUF",
	ModelRevision:         "fc034cfff751157913579611efad8462ac1be606",
	ModelFile:             "gemma-4-12b-it-UD-Q4_K_XL.gguf",
	ModelSHA256:           "90fd944d227e9d9b68e7e2c7d5b57b79d4c66ed521b0919fbbd932cf834f6f8e",
	ModelBytes:            7366423360,
	ModelLicense:          "Apache-2.0",
	TokenizerRepository:   "google/gemma-4-12b-it",
	TokenizerRevision:     "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
	TokenizerSHA256:       "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
	TokenizerConfigSHA256: "a62f4e85a47c0c136edaaa3a4f591fd6783717299a9def47e5ad03a49f6a5eb9",
	ChatTemplateSHA256:    "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
	LlamaRepository:       "https://github.com/thecodacus/llama.cpp",
	LlamaRevision:         "14d04e755fa28653e87b9a07072892265bdc0fad",
	LlamaLicense:          "MIT",
	RLCDRepository:        "https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD",
	RLCDRevision:          "2af86848be75847ccb3553b0941cc51d6ef7e4e9",
	RLCDLicense:           "Apache-2.0",
	PlaygroundRepository:  "https://github.com/thecodacus/decision-playground",
	PlaygroundRevision:    "843f72e61f5ebaa1c2225c4902849bd42af44c05",
	PlaygroundLicense:     "none; behavior reference only",
}
