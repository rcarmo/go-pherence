# MiniCPM-V/O and Kev retirement

The owner retired the MiniCPM-V/O and Kev work on 27 September 2026. `model/minicpmv`, its config parsers, inspector, fixtures, asset helpers and Make targets were removed. Kev had no `model/kev` implementation in this repository; its porting roadmap and open plan items were removed. No released-model MiniCPM-V/O parity or end-to-end generation had qualified, and Kev-specific pointer-head execution had not begun.

Shared Qwen, Mistral, Whisper, SigLIP and other runtime components remain because they have independent consumers. The model-coverage manifest, downloader and current support matrix no longer advertise MiniCPM-V/O. Historical validation reports remain in Git history and in `docs/validation/` where applicable; commands pointing to removed source are historical, not current instructions. This retirement is separate from [MoJev's removal](mojev-retirement-20260927.md) and the continuing Needle 3 work.
