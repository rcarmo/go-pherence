# backends/spacemit/k3engine/config

`k3engine` feature flags, all driven by `IME2_*` environment variables and read
once at process start. They select kernel paths, fusion, TCM staging, and
debug/compare modes.

Examples: `Q4kScaledLoopOn` (`IME2_Q4K_SCALED_LOOP`), `Q4kGateFuseOn`,
`Q4kFFNFuseOn`, `Q4kTCMBWaveOn`, `NativeI8I8On`, `Q8RoundOn`, `Q4kCShimOn`.

Pure leaf package (only `os`); imported by `k3engine`.
