#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from omnivoice.utils.audio import fade_and_pad_audio, remove_silence

SR = 24000


def tone(ms: int, amp: float) -> np.ndarray:
    return np.full((1, ms * SR // 1000), amp, dtype=np.float32)


def silence(ms: int) -> np.ndarray:
    return np.zeros((1, ms * SR // 1000), dtype=np.float32)


def vec(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts, axis=1)


CASES = {
    "short_edges": vec(silence(120), tone(80, 0.5), silence(140)),
    "long_middle": vec(tone(100, 0.5), silence(900), tone(100, -0.5)),
    "middle_preserved": vec(tone(100, 0.5), silence(350), tone(100, 0.25)),
    "low_amplitude": tone(1000, 0.001),
    "all_silent": silence(1000),
}

FADE_INPUT = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = {
        "sample_rate": SR,
        "silence_cases": [],
        "fade_case": {
            "input": FADE_INPUT[0].tolist(),
            "custom": fade_and_pad_audio(
                FADE_INPUT,
                pad_duration=1 / SR,
                fade_duration=2 / SR,
                sample_rate=SR,
            )[0].tolist(),
        },
    }

    for name, audio in CASES.items():
        payload["silence_cases"].append(
            {
                "name": name,
                "input": audio[0].tolist(),
                "default": remove_silence(audio, SR)[0].tolist(),
                "reference": remove_silence(audio, SR, mid_sil=200, lead_sil=100, trail_sil=200)[0].tolist(),
            }
        )

    Path(args.output).write_text(json.dumps(payload))


if __name__ == "__main__":
    main()
