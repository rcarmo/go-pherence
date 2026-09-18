#!/usr/bin/env python3
"""Time unchanged upstream iterative generation using identical prepared Go inputs.
Requires upstream OmniVoice/PyTorch environment; Python is needed for this baseline.
Only _prepare_inference_inputs is replaced; upstream batching, schedule and sampling
remain unchanged. No reference encoding or text tokenisation is timed.
"""
import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig, GenerationTask

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--prompt', required=True)
ap.add_argument('--output', required=True)
ap.add_argument('--steps', type=int, default=8)
ap.add_argument('--threads', type=int, default=2)
a = ap.parse_args()
if a.steps < 1 or a.steps > 128 or a.threads < 1:
    raise ValueError('invalid steps/threads')
if Path(a.output).exists():
    raise FileExistsError(a.output)
p = json.loads(Path(a.prompt).read_text())
torch.set_num_threads(a.threads)
torch.set_num_interop_threads(1)
torch.manual_seed(42)
start = time.perf_counter()
m = OmniVoice.from_pretrained(a.model, device_map='cpu', dtype=torch.float32,
                              local_files_only=True).eval()
loaded = time.perf_counter()
books = m.config.num_audio_codebook
cond, uncond = p['conditional'], p['unconditional']
target = p['target_frames']
ids = torch.tensor(cond['ids'], dtype=torch.long).reshape(1, books, cond['tokens'])
audio = torch.tensor(cond['audio_mask'], dtype=torch.bool).reshape(1, cond['tokens'])
uids = torch.tensor(uncond['ids'], dtype=torch.long).reshape(1, books, uncond['tokens'])
uaudio = torch.tensor(uncond['audio_mask'], dtype=torch.bool).reshape(1, uncond['tokens'])
assert uncond['tokens'] == target
assert torch.equal(uids, ids[:, :, -target:])
assert torch.equal(uaudio, audio[:, -target:]) and bool(uaudio.all())
assert bool((uids == m.config.audio_mask_id).all())
assert not cond.get('positions') and not cond.get('mask')
assert not uncond.get('positions') and not uncond.get('mask')
# Upstream still assembles and pads its own batch, as in normal generation.
def prepared(*args, **kwargs):
    return {'input_ids': ids, 'audio_mask': audio}
m._prepare_inference_inputs = prepared
cfg = OmniVoiceGenerationConfig(num_step=a.steps, guidance_scale=2.0,
    t_shift=0.1, layer_penalty_factor=5.0, class_temperature=0.0,
    position_temperature=5.0, preprocess_prompt=False, postprocess_output=False)
task = GenerationTask(batch_size=1, texts=[p['text']], target_lens=[target],
    langs=[None], instructs=[None], ref_texts=[None], ref_audio_tokens=[None],
    ref_rms=[p.get('ref_rms')])
prepared_at = time.perf_counter()
with torch.inference_mode():
    codes = m._generate_iterative(task, cfg)[0]
    generated = time.perf_counter()
    wave = m.audio_tokenizer.decode(codes.unsqueeze(0)).audio_values.flatten().float()
    decoded = time.perf_counter()
report = {
    'upstream_loop_unmodified': True,
    'prepared_prompt_sha256': hashlib.sha256(Path(a.prompt).read_bytes()).hexdigest(),
    'torch_version': torch.__version__, 'attention_implementation': m.llm.config._attn_implementation,
    'threads': a.threads, 'interop_threads': 1, 'steps': a.steps, 'guidance': 2,
    'seed': 42, 'rng': 'PyTorch; not Go PCG', 'conditional_tokens': cond['tokens'],
    'unconditional_tokens': target, 'batch_shape': [2, books, cond['tokens']],
    'load_seconds': loaded-start, 'prepared_setup_seconds': prepared_at-loaded,
    'denoise_seconds': generated-prepared_at, 'decode_seconds': decoded-generated,
    'inference_seconds': decoded-prepared_at, 'total_seconds': decoded-start,
    'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    'samples': wave.numel(), 'wave_finite': bool(torch.isfinite(wave).all()),
    'code_sha256': hashlib.sha256(codes.cpu().numpy().tobytes()).hexdigest(),
    'notes': ['No waveform postprocessing or file encoding in timed inference.',
              'No reference/text preparation; same saved IDs as Go prepared generation.',
              'Same seed does not imply same noise or audio across runtimes.']
}
with open(a.output, 'x') as f:
    json.dump(report, f, indent=2)
    f.write('\n')
print(json.dumps(report))
