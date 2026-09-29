"""Independent released-weight 13,000-sample JFK cache-aware generation encoder fixture.

Generation requires 25 mel frames first, then 32, with no attention_mask.
Processor's 26th first frame is masked and discarded; final mel rows are
zero-padded to 32 without passing a mask, as the pinned generate path does.
This checks the encoder and prompt projection; it is not text/WER evidence.
"""
import argparse
import gzip
import json
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForRNNT, AutoProcessor
from transformers.cache_utils import DynamicCache
from transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming import NemotronAsrStreamingEncoderCausalConvPaddingCache

ROOT = Path(__file__).resolve().parents[1]


def save(path, tensor):
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    with path.open("wb") as file:
        with gzip.GzipFile(filename="", fileobj=file, mode="wb", mtime=0) as compressed:
            compressed.write(array.tobytes())
    return list(array.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    audio, rate = sf.read(ROOT / "testdata/jfk.wav", dtype="float32")
    if rate != 16000 or len(audio) != 176000:
        raise ValueError("unexpected JFK audio")
    processor = AutoProcessor.from_pretrained(ROOT / "checkpoints/nemotron/asr", local_files_only=True)
    model = AutoModelForRNNT.from_pretrained(ROOT / "checkpoints/nemotron/asr", local_files_only=True).eval()
    first = processor(audio[:4040], sampling_rate=rate, is_streaming=True,
                      is_first_audio_chunk=True, return_tensors="pt")
    middle = processor(audio[3744:9264], sampling_rate=rate, is_streaming=True,
                       is_first_audio_chunk=False, return_tensors="pt")
    terminal_audio = torch.cat([torch.from_numpy(audio[8864:13000]), torch.zeros(56)]).numpy()
    last = processor(terminal_audio, sampling_rate=rate, is_streaming=True,
                     is_first_audio_chunk=False, return_tensors="pt")
    if [list(x.input_features.shape) for x in (first, middle, last)] != [[1, 26, 128], [1, 32, 128], [1, 24, 128]]:
        raise ValueError("unexpected processor chunk sizes")
    if [int(x.attention_mask.sum()) for x in (first, middle, last)] != [25, 32, 24]:
        raise ValueError("unexpected processor masks")
    chunks = (first.input_features[:, :25], middle.input_features,
              torch.cat([last.input_features, torch.zeros(1, 8, 128)], dim=1))
    kv = DynamicCache(config=model.encoder.config)
    conv = NemotronAsrStreamingEncoderCausalConvPaddingCache()
    independent_sub = NemotronAsrStreamingEncoderCausalConvPaddingCache()
    inputs, towers, encoded, layer0 = [], [], [], []
    def capture_layer0(module, args, output):
        layer0.append(output[0].detach().cpu())
    hook = model.encoder.layers[0].register_forward_hook(capture_layer0)
    with torch.inference_mode():
        for index, chunk in enumerate(chunks):
            # Do not pass a 2D attention mask: streaming generate calls
            # get_audio_features without one, including on the final chunk.
            projected = model.encoder.subsampling(chunk, padding_cache=independent_sub) * model.encoder.input_scale
            result = model.encoder(input_features=chunk, past_key_values=kv,
                                   padding_cache=conv, use_cache=True, num_lookahead_tokens=3)
            if projected.shape != (1, 4, 1024) or result.last_hidden_state.shape != (1, 4, 1024):
                raise ValueError("unexpected cached encoder shape")
            if kv.get_seq_length() != (index + 1) * 4:
                raise ValueError("cache did not advance")
            prompt = torch.nn.functional.one_hot(torch.tensor([model.config.default_prompt_id]),
                                                  num_classes=model.config.num_prompts).to(chunk.dtype)
            fused = model.prompt_projector(torch.cat([result.last_hidden_state,
                prompt[:, None, :].expand(-1, 4, -1)], dim=-1))
            inputs.append(projected[0].cpu())
            towers.append(result.last_hidden_state[0].cpu())
            encoded.append(model.encoder_projector(fused)[0].cpu())
    hook.remove()
    with torch.inference_mode():
        generated = model.generate(input_features=(chunk for chunk in chunks),
                                   num_lookahead_tokens=3, max_new_tokens=100)
    tokens = generated.sequences[0].tolist()
    if tokens and tokens[0] == model.config.blank_token_id:
        tokens = tokens[1:]  # generation includes its initial BOS blank
    shapes = {
        "generated_tokens": tokens,
        "input": save(args.out / "input.f32.gz", torch.cat(inputs)),
        "layer0": save(args.out / "layer0.f32.gz", torch.cat(layer0)),
        "tower": save(args.out / "tower.f32.gz", torch.cat(towers)),
        "rnnt": save(args.out / "rnnt.f32.gz", torch.cat(encoded)),
        "chunks": [list(x.shape) for x in chunks],
    }
    print(json.dumps(shapes))


if __name__ == "__main__":
    main()
