# Nemotron AMI ES2004a 100-second follow-up — 30 September 2026

All six task/backend runs on a 100-second natural-speech AMI excerpt completed faster than audio duration on the local i7-12700/RTX 3060. This interval extends the [350–410-second pilot](nemotron-pilot-20260930.md) to 350–450 seconds and therefore **shares its first 60 seconds**. The two quality measurements are correlated, not independent corpus samples.

The official AMI CC BY 4.0 headset-mix WAV and annotation ZIP have the source hashes in [`source.json`](source.json). Run `scripts/prepare_ami_corpus.py` with those pinned sources, `--meeting ES2004a --start 350 --duration 100`, and a new ignored output directory. The locally generated manifest recorded a canonical WAV with 1,600,000 samples and SHA-256 `47a1a3569aac9dcf9c33e069d65045a151aa37358ae57c816609c1356578395b`; RTTM SHA-256 `dc158bd486471e24970308a109b1cf66baab1a05286b342710eb26b0f531785c` (39 turns); timed-word JSON SHA-256 `87238f284c786962a5ce2ebc5cc6b739da706867ee1234239538e9f95a5f4197` (237 words). Keep the generated WAV, RTTM, word JSON and source media out of Git.

The binary was built from clean `ede9ea019fad6732962c9dddc49578fc1add9c90` using `go build -o /workspace/tmp/nemotron-ami-es2004a-350-410/nemotron-cli ./cmd/audio/nemotron`. For every task/backend, the command below measured full-process wall time (excluding compilation, including WAV and checkpoint loading), using `GOMAXPROCS=4`:

```sh
p=/workspace/tmp/nemotron-ami-es2004a-350-450
cli=/workspace/tmp/nemotron-ami-es2004a-350-410/nemotron-cli
for backend in simd ptx vulkan; do
  for task in asr diarization; do
    model=checkpoints/nemotron/asr/model.safetensors
    extension=txt
    tower=
    if [ "$task" = diarization ]; then
      model=checkpoints/nemotron/diarization/model.safetensors
      extension=json
      case "$backend" in ptx) tower=-ptx-tower;; vulkan) tower=-vulkan-tower;; esac
    fi
    GOMAXPROCS=4 /usr/bin/time -f 'wall_elapsed=%e maxrss_kb=%M' \
      -o "$p/$backend-$task.time" "$cli" -task "$task" -backend "$backend" $tower \
      -input "$p/ES2004a-350-450.wav" -model "$model" \
      > "$p/$backend-$task.$extension" 2> "$p/$backend-$task.log"
  done
done
```

| Backend | ASR wall / peak RSS | Diarization wall / peak RSS |
|---|---:|---:|
| SIMD | 59.88 s / 5,175,536 KiB | 74.53 s / 1,587,840 KiB |
| PTX | 58.44 s / 5,295,152 KiB | 24.21 s / 1,718,012 KiB |
| Vulkan | 59.95 s / 5,350,264 KiB | 30.93 s / 2,738,120 KiB |

CLI request times excluding loading were respectively 57.68, 56.23 and 57.71 seconds for ASR; 74.02, 23.65 and 30.40 seconds for diarization. All six runs used the same local checkpoint and tokenizer pins recorded in the 60-second report. The ASR PTX/Vulkan paths offload only subsampling projection. Diarization PTX/Vulkan used the opt-in resident tower; frontend, audio layer 0, head and speaker cache still run on CPU. One sample per case cannot quantify timing variation or sustained throughput.

All three saved transcripts contained 166 whitespace tokens and all three saved diarization outputs contained 58 speaker spans with the same parsed geometry on this run. Transcript file SHA-256 `ab692003c91ffee919f9ab371a480d8cefa2b0d6248c61cc1518edf076abde7a`; span file SHA-256 `c6c64cf9d80cfb7691aef1f013dec585b05a11673b0d94080e58beb7b671e5d1`. These identify saved outputs, not a future byte-exact numerical gate.

A local scoring pass reused `score_speaker_words.py` NFKC/casefold tokenisation and edit counts, and `score_community1_corpus.py` pyannote DER/JER conventions: UEM `[0,100]`, overlap included (`skip_overlap=False`), collars 0 and 0.25 seconds. This pass used external `pyannote.metrics==4.1` / `pyannote.database==6.1.1`; it did not modify project dependencies. Against 237 independent reference tokens and 166 hypothesis tokens, transcript-only WER was 83/237 = **0.350211** (10 substitutions, 72 deletions, 1 insertion). At zero collar, DER was **0.511447** (52.236 missed speaker-seconds, 4.553 confused, zero false alarms against 111.036 reference speaker-seconds) and JER **0.565462**. At a 0.25-second collar, DER was **0.490473** and JER **0.558721**. These are absolute scores for one excerpt with no timed hypothesis words, speaker-attributed WER or pinned independent reference-system result. The 60-second pilot's reusable scorer is intentionally bound to its committed 350–410-second asset contract; this follow-up has not yet been promoted to a multi-case scoring contract.

An isolated ASR SGEMM K-block experiment (128 to 64) increased latency across the measured five-row matrix shapes and was reverted without a commit. The one-load warm JFK 11-second ASR benchmark gave five five-iteration samples of 6.37–6.88 seconds per request, and the 30-second diarization SIMD benchmark completed four samples near 10.96–12.22 seconds per request before a 300-second harness timeout during the fifth sample; a separate four-iteration run passed at 11.02 seconds per request. These smaller/repeated-input benchmarks do not replace natural 100-second or long-lived stream soak evidence.
