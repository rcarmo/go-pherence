"""Pin pre-compression speaker-cache FIFO transitions from released Transformers.

Uses deterministic prepared 512-wide encoder embeddings and 8-speaker logits.
These are cache-operator inputs, not a diarization model's streaming output.
"""
import argparse
import gzip
from pathlib import Path

import torch
from transformers import AutoModelForAudioFrameClassification
from transformers.models.nemotron3_diarization.modeling_nemotron3_diarization import Nemotron3DiarizationSpeakerCache

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "checkpoints/nemotron/diarization"


def save(path, tensor):
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    with path.open("wb") as out:
        with gzip.GzipFile(filename="", fileobj=out, mode="wb", mtime=0) as compressed:
            compressed.write(array.tobytes())
    print(path.name, array.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForAudioFrameClassification.from_pretrained(MODEL, local_files_only=True).eval()
    with torch.inference_mode():
        cache = Nemotron3DiarizationSpeakerCache(model.config.streaming_config)
        if (cache.fifo_length, cache.speaker_cache_update_period, cache.speaker_cache_length) != (264, 222, 264):
            raise ValueError("streaming cache config changed")
        # 9,4 is the released low-latency stream mode. The prepared chunks
        # below include four right-context frames, which must not enter FIFO.
        for index, frames in enumerate((9, 9, 9)):
            lookahead = 4
            chunk = torch.arange(index * 13, index * 13 + frames + lookahead, dtype=torch.float32)
            embeds = ((chunk[:, None] * 7 + torch.arange(512)[None, :]) % 251 - 125) / 127
            embeds = embeds[None]
            cached = cache.get_embeds(embeds)
            full = torch.cat([cached, embeds], dim=1)
            save(args.out / f"cache_step{index}_input.f32.gz", full[0])
            logits = torch.arange(full.shape[1] * 8 * 8, dtype=torch.float32).reshape(1, full.shape[1] * 8, 8)
            logits = ((logits % 53) - 26) / 11
            mask = torch.ones(1, full.shape[1], dtype=torch.bool)
            if index == 2:
                mask[0, -1] = False
            cache.update(full, logits, model.silence_embeds, frames, mask=mask)
            save(args.out / f"cache_step{index}_fifo.f32.gz", cache.fifo[0, :cache.num_fifo_frames])
            save(args.out / f"cache_step{index}_pooled.f32.gz", cache._pool_probs(logits, mask)[0])
            print("step", index, "cache", cache.num_cache_frames, "fifo", cache.num_fifo_frames)


if __name__ == "__main__":
    main()
