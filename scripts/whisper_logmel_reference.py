#!/usr/bin/env python3
"""Generate a tiny synthetic 128-band reference without loading models or torch.

SCRIPT_JDOC: {"summary":"Generate synthetic Whisper 128-bin NumPy reference fixtures from checksum-pinned Transformers audio utilities","kind":"mutating","weight":"lightweight","role":"entrypoint"}

Requires NumPy and an explicitly supplied copy of Transformers v4.57.1
src/transformers/audio_utils.py. Only the reviewed numerical function definitions
are loaded. Never imports Transformers or downloads weights. CPU fixture work
only. The Go implementation is not used to generate expected values.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Optional, Union
import warnings

import numpy as np

SOURCE = 'https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/audio_utils.py'
# Hash verified against the downloaded tagged source before fixture generation.
SHA256 = 'c038450307a8dbc9a95eed8e413770f34814fa38058121ca9f2eee8dd0dfa958'
FUNCTIONS = {'hertz_to_mel', 'mel_to_hertz', '_create_triangular_filter_bank',
             'mel_filter_bank', 'window_function', 'spectrogram'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    raw = args.reference.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise SystemExit('Reference checksum mismatch; do not generate a fixture from different code.')
    tree = ast.parse(raw)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    if {node.name for node in functions} != FUNCTIONS:
        raise SystemExit('Missing reference functions')
    namespace = {'np': np, 'Optional': Optional, 'Union': Union, 'warnings': warnings}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(args.reference), 'exec'), namespace)
    mel_filters = namespace['mel_filter_bank'](num_frequency_bins=201, num_mel_filters=128,
        min_frequency=0.0, max_frequency=8000.0, sampling_rate=16000,
        norm='slaney', mel_scale='slaney')
    window = namespace['window_function'](400, 'hann')
    cases = []
    for name, length, style in [('broadband', 1600, 'lcg'), ('odd-boundary', 1001, 'impulse'),
                                ('short-reflect', 160, 'lcg'), ('silence', 320, 'silence')]:
        state = 0x12345678
        values = []
        for i in range(length):
            state = (1664525 * state + 1013904223) & 0xffffffff
            value = ((state >> 16) - 32768) / 65536 if style == 'lcg' else 0
            if style == 'impulse' and i in (0, 199, length-1):
                value = (0.75 if i != 199 else -0.25)
            values.append(value)
        samples = np.array(values, dtype=np.float32)
        # WhisperFeatureExtractor._np_extract_fbank_features (v4.57.1), including
        # final-frame removal and its float32 clamp/normalisation boundary.
        spectrum = namespace['spectrogram'](samples, window, frame_length=400,
            hop_length=160, power=2.0, mel_filters=mel_filters, log_mel='log10')[:, :-1]
        spectrum = (np.maximum(spectrum, spectrum.max()-8.0)+4.0)/4.0
        cases.append({'name': name, 'samples': samples.tolist(), 'shape': list(spectrum.shape),
                      'mel': spectrum.reshape(-1).tolist()})
    report = {'reference': SOURCE, 'sha256': SHA256, 'transformers_version': '4.57.1',
              'numpy_version': np.__version__, 'license': 'Synthetic fixture generated locally; reference code Apache-2.0',
              'algorithm': 'WhisperFeatureExtractor._np_extract_fbank_features; 128 Slaney bins, FFT400, hop160',
              'cases': cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Wrote {len(cases)} synthetic fixtures to {args.output}')


if __name__ == '__main__':
    main()
