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
        # At the first overflow, 222 oldest FIFO frames move into the
        # uncompressed speaker cache. A following chunk checks that the
        # retained speaker/FIFO state is prepended in the reference order.
        # Isolate score policy before top-k compression: deterministic,
        # non-tied probabilities exercise speech, silence and mask branches.
        score_rows = torch.arange(300, dtype=torch.float32)[:, None]
        score_speakers = torch.arange(8, dtype=torch.float32)[None, :]
        score_probs = 0.05 + 0.9 * ((score_rows * 19 + score_speakers * 37) % 299) / 299
        score_probs[0] = 0.1
        score_probs[1] = 0.51
        score_probs[2] = 0.9
        save(args.out / "cache_score_probs.f32.gz", score_probs)
        save(args.out / "cache_score_output.f32.gz", cache._get_frame_scores(score_probs[None])[0])
        speech_probs = torch.full((300, 8), 0.05)
        speech_probs[::3, 0] = 0.95
        speech_probs[::5, 1] = 0.92
        save(args.out / "cache_score_speech_probs.f32.gz", speech_probs)
        save(args.out / "cache_score_speech_output.f32.gz", cache._get_frame_scores(speech_probs[None])[0])
        for label in ("pattern", "sweep"):
            frame_ids = torch.arange(486, dtype=torch.float32)
            compress_probs = torch.full((486, 8), 0.05)
            if label == "pattern":
                compress_probs[::3, 0] = 0.95
                compress_probs[::5, 1] = 0.92
            else:
                for speaker in range(8):
                    compress_probs[:, speaker] = 0.05 + 0.9 * ((frame_ids * (19 + speaker) + speaker * 37) % 487) / 487
            compress_embeds = torch.arange(486 * 512, dtype=torch.float32).reshape(486, 512) / 10000
            save(args.out / f"cache_compress_{label}_probs.f32.gz", compress_probs)
            output_embeds, output_probs = cache._compress(compress_embeds[None], compress_probs[None], model.silence_embeds)
            save(args.out / f"cache_compress_{label}_embeds.f32.gz", output_embeds[0])
            save(args.out / f"cache_compress_{label}_selected_probs.f32.gz", output_probs[0])
        for index, frames in enumerate((237, 9), start=3):
            lookahead = 4
            chunk = torch.arange(index * 241, index * 241 + frames + lookahead, dtype=torch.float32)
            embeds = ((chunk[:, None] * 7 + torch.arange(512)[None, :]) % 251 - 125) / 127
            embeds = embeds[None]
            cached = cache.get_embeds(embeds)
            full = torch.cat([cached, embeds], dim=1)
            save(args.out / f"cache_step{index}_input.f32.gz", full[0])
            logits = torch.arange(full.shape[1] * 8 * 8, dtype=torch.float32).reshape(1, full.shape[1] * 8, 8)
            logits = ((logits % 53) - 26) / 11
            mask = torch.ones(1, full.shape[1], dtype=torch.bool)
            cache.update(full, logits, model.silence_embeds, frames, mask=mask)
            save(args.out / f"cache_step{index}_fifo.f32.gz", cache.fifo[0, :cache.num_fifo_frames])
            save(args.out / f"cache_step{index}_speaker.f32.gz", cache.embeds[0, :cache.num_cache_frames])
            save(args.out / f"cache_step{index}_speaker_probs.f32.gz", cache.probs[0, :cache.num_cache_frames])
            print("step", index, "cache", cache.num_cache_frames, "fifo", cache.num_fifo_frames, "compressed", cache.is_compressed)
        # Step 5 first exceeds the 264-frame speaker cache. Its 222-frame
        # transfer invokes score-based compression. Step 6 checks the
        # subsequent prepared input uses the compressed cache order.
        for index, frames in enumerate((222, 9), start=5):
            lookahead = 4
            chunk = torch.arange(index * 241, index * 241 + frames + lookahead, dtype=torch.float32)
            embeds = ((chunk[:, None] * 7 + torch.arange(512)[None, :]) % 251 - 125) / 127
            embeds = embeds[None]
            cached = cache.get_embeds(embeds)
            full = torch.cat([cached, embeds], dim=1)
            save(args.out / f"cache_step{index}_input.f32.gz", full[0])
            logits = torch.arange(full.shape[1] * 8 * 8, dtype=torch.float32).reshape(1, full.shape[1] * 8, 8)
            logits = ((logits % 53) - 26) / 11
            # Avoid the modulo pattern's top-k cutoff ties: PyTorch does not
            # promise which equal-score frame survives the selection.
            logits += torch.arange(full.shape[1], dtype=torch.float32)[None, :, None].repeat_interleave(8, dim=1) * 0.00017
            logits += torch.arange(8, dtype=torch.float32)[None, None, :] * 0.000037
            mask = torch.ones(1, full.shape[1], dtype=torch.bool)
            cache.update(full, logits, model.silence_embeds, frames, mask=mask)
            save(args.out / f"cache_step{index}_fifo.f32.gz", cache.fifo[0, :cache.num_fifo_frames])
            save(args.out / f"cache_step{index}_speaker.f32.gz", cache.embeds[0, :cache.num_cache_frames])
            save(args.out / f"cache_step{index}_speaker_probs.f32.gz", cache.probs[0, :cache.num_cache_frames])
            print("step", index, "cache", cache.num_cache_frames, "fifo", cache.num_fifo_frames, "compressed", cache.is_compressed)

if __name__ == "__main__":
    main()
