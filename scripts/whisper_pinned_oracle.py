#!/usr/bin/env python3
"""Offline reference comparison only; never imported by Go runtime.

Requires isolated Transformers 4.57.1, CPU PyTorch, and pre-exported public PCM /
Go tensors from TestSpeechOracleExport. No downloads or credentials. Output goes
into a fresh caller-specified JSON file; weights and canonical WAV stay external.
"""
import argparse
import copy
import hashlib
import json
import pathlib
import wave
import subprocess
import tokenizers
import numpy as np
import torch
import transformers
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor
from transformers.generation.logits_process import WhisperTimeStampLogitsProcessor, SuppressTokensLogitsProcessor, SuppressTokensAtBeginLogitsProcessor
from transformers.modeling_outputs import BaseModelOutput
from tokenizers import Tokenizer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def difference(a, b):
    if a.shape != b.shape:
        raise ValueError((a.shape, b.shape))
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite tensors')
    error = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return dict(shape=list(a.shape), max_abs=float(error.max()), mean_abs=float(error.mean()),
                rms=float(np.sqrt(np.mean(error ** 2))), p99_abs=float(np.quantile(error, .99)))


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def timestamp_segments(ids, tokenizer):
    begin = tokenizer.token_to_id('<|0.00|>')
    eot = tokenizer.token_to_id('<|endoftext|>')
    start, text, segments = None, [], []
    for token in ids:
        if token == eot:
            if text:
                raise ValueError('unclosed reference timestamp')
            break
        if token >= begin:
            end = (token-begin)/50
            if text:
                if start is None or end <= start:
                    raise ValueError('invalid reference timestamp span')
                segments.append(dict(Start=start, End=end, Text=tokenizer.decode(text, skip_special_tokens=False).strip(), Tokens=text))
                text = []
            start = end
        else:
            if start is None:
                raise ValueError('reference text before timestamp')
            text.append(token)
    if not ids or ids[-1] != eot or text:
        raise ValueError('reference did not finish cleanly with EOT')
    return segments


def cached_logits(model, encoder, ids):
    past, logits = None, []
    for token in ids:
        out = model(encoder_outputs=BaseModelOutput(last_hidden_state=encoder),
                    decoder_input_ids=torch.tensor([[token]]), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits.append(out.logits[0,-1].numpy().copy())
    return np.stack(logits)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=pathlib.Path)
    p.add_argument('--exports', required=True, type=pathlib.Path)
    p.add_argument('--output', required=True, type=pathlib.Path)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('output must be new')
    if transformers.__version__ != '4.57.1' or torch.__version__ != '2.14.0+cpu' or tokenizers.__version__ != '0.22.2' or np.__version__ != '2.5.3':
        raise ValueError('requires recorded exact Transformers/Torch/tokenizers/NumPy versions')
    hashes = {'model.safetensors':'7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395',
              'config.json':'ffdccec4f3211f4c63310f2b7098f309fe70f3952cedc5e4d11e43f5b2379b98',
              'generation_config.json':'a5d5325911f16e74001a72fa13d6e208eee51548f994646de1f4b4cc8b35b512',
              'tokenizer.json':'27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566',
              'preprocessor_config.json':'9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666'}
    for filename, expected in hashes.items():
        if digest(args.model / filename) != expected:
            raise ValueError('model asset hash ' + filename)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = WhisperForConditionalGeneration.from_pretrained(str(args.model), local_files_only=True,
               dtype=torch.float32, attn_implementation='eager').eval()
    extractor = WhisperFeatureExtractor.from_pretrained(str(args.model), local_files_only=True)
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    repo = pathlib.Path(__file__).resolve().parents[1]
    pinned_policy = json.loads((args.model/'generation_config.json').read_text())
    report = dict(transformers=transformers.__version__, torch=torch.__version__, numpy=np.__version__, tokenizers=tokenizers.__version__,
                  threads=2, dtype='float32', attention='eager', model_hashes=hashes,
                  script_sha256=digest(pathlib.Path(__file__)), go_export_test_sha256=digest(repo/'models/whisper/oracle_export_test.go'),
                  go_base_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip(),
                  runtime_integration=False, fixtures=[])
    with torch.inference_mode():
        for directory in sorted(args.exports.iterdir()):
            if not directory.is_dir():
                continue
            meta = json.loads((directory / 'metadata.json').read_text())
            with wave.open(str(directory/'canonical.wav'), 'rb') as f:
                assert f.getnchannels() == 1 and f.getframerate() == 16000 and f.getsampwidth() == 2
                pcm = np.frombuffer(f.readframes(f.getnframes()), dtype='<i2').astype(np.float32) / 32768
            assert len(pcm) == meta['pcm_samples']
            features = extractor(pcm, sampling_rate=16000, return_tensors='pt').input_features
            go_features = np.fromfile(directory/'mel.f32', dtype='<f4').reshape(meta['mel_shape'])
            go_encoder = np.fromfile(directory/'encoder.f32', dtype='<f4').reshape(meta['encoder_shape'])
            encoder = model.model.encoder(features).last_hidden_state
            encoder_on_go = model.model.encoder(torch.from_numpy(go_features.copy())[None]).last_hidden_state
            inputs = torch.tensor([meta['decoder_inputs']])
            logits = model(encoder_outputs=BaseModelOutput(last_hidden_state=encoder), decoder_input_ids=inputs, use_cache=False).logits[0].numpy()
            logits_on_go = model(encoder_outputs=BaseModelOutput(last_hidden_state=torch.from_numpy(go_encoder.copy())[None]), decoder_input_ids=inputs, use_cache=False).logits[0].numpy()
            go_logits = np.fromfile(directory/'logits.f32', dtype='<f4').reshape(meta['logits_shape'])
            item = dict(fixture=meta['fixture'], language=meta['language'], reference=meta['reference'],
                        mel=difference(features[0].numpy(),go_features),
                        encoder=difference(encoder[0].numpy(),go_encoder),
                        encoder_same_go_features=difference(encoder_on_go[0].numpy(),go_encoder),
                        logits_on_go_token_prefix=difference(logits,go_logits),
                        logits_same_go_encoder=difference(logits_on_go,go_logits),
                        cached_logits_on_go_prefix=difference(cached_logits(model, encoder, meta['decoder_inputs']), go_logits),
                        cached_logits_same_go_encoder=difference(cached_logits(model, torch.from_numpy(go_encoder.copy())[None], meta['decoder_inputs']), go_logits),
                        input_hashes={name:digest(directory/name) for name in ['canonical.wav','metadata.json','mel.f32','encoder.f32','logits.f32']})
            # Use pinned model policy as the reference source, not exported Go
            # metadata; equality here checks the Go policy resolution as well.
            if meta['suppress'] != pinned_policy['suppress_tokens'] or meta['begin_suppress'] != pinned_policy['begin_suppress_tokens'] or meta['max_initial_timestamp_index'] != pinned_policy['max_initial_timestamp_index']:
                raise ValueError('Go generation policy differs from pinned reference')
            cfg = copy.deepcopy(model.generation_config)
            cfg.no_timestamps_token_id = tokenizer.token_to_id('<|notimestamps|>')
            cfg.max_initial_timestamp_index = pinned_policy['max_initial_timestamp_index']
            cfg.forced_decoder_ids = None
            item['decode'] = {}
            for mode, enc in [('reference', encoder), ('same_go_encoder', torch.from_numpy(go_encoder.copy())[None])]:
                ids = [tokenizer.token_to_id(x) for x in ['<|startoftranscript|>', '<|'+meta['language']+'|>', '<|transcribe|>']]
                if ids != meta['decoder_inputs'][:meta['prompt_length']]:
                    raise ValueError('prompt mismatch')
                processor = WhisperTimeStampLogitsProcessor(cfg, begin_index=len(ids))
                suppress = SuppressTokensLogitsProcessor(pinned_policy['suppress_tokens'], device='cpu')
                begin = SuppressTokensAtBeginLogitsProcessor(pinned_policy['begin_suppress_tokens'], begin_index=len(ids), device='cpu')
                generated, past = [], None
                for i in range(96):
                    tokens = torch.tensor([ids if past is None else [ids[-1]]])
                    out = model(encoder_outputs=BaseModelOutput(last_hidden_state=enc), decoder_input_ids=tokens, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    scores = out.logits[:,-1,:].clone()
                    prefix = torch.tensor([ids])
                    scores = processor(prefix, begin(prefix, suppress(prefix, scores)))
                    token = int(scores.argmax(dim=-1))
                    generated.append(token)
                    if token == meta['eot']:
                        break
                    ids.append(token)
                segments = timestamp_segments(generated, tokenizer)
                item['decode'][mode] = dict(generated=generated, segments=segments,
                    text=' '.join(s['Text'] for s in segments), eot=generated[-1] == meta['eot'],
                    first_token_diff=first_diff(generated, meta['go_generated']),
                    generated_matches_go=generated == meta['go_generated'], segments_match_go=segments == meta['go_segments'])
            item['go_text'] = ' '.join(s['Text'] for s in meta['go_segments'])
            item['go_generated'] = meta['go_generated']
            item['go_segments'] = meta['go_segments']
            report['fixtures'].append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    expected = {'jfk', 'minds-fr-0', 'minds-pt-0', 'minds-pt-1', 'silence-5s'}
    names = [f['fixture'] for f in report['fixtures']]
    complete = len(names) == len(expected) and set(names) == expected
    report['token_segment_parity_pass'] = complete and all(
        d['eot'] and d['generated_matches_go'] and d['segments_match_go']
        for f in report['fixtures'] for d in f['decode'].values())
    report['numerical_metrics_are_diagnostics'] = True
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    if not report['token_segment_parity_pass']:
        raise SystemExit('FAIL: fixture completeness or token/segment parity; report preserved')

if __name__ == '__main__':
    main()
