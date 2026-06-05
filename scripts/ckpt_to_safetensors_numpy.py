#!/usr/bin/env python3
"""Torch-free SpeechBrain ECAPA .ckpt -> safetensors converter.

Reads a torch.save (zip+pickle) checkpoint using only numpy + the stdlib, and
writes a safetensors file preserving the original float tensor names (the layout
models/speaker.LoadSpeechBrainECAPASafetensors expects). Used on hosts without
torch/safetensors (e.g. the RISC-V board).

    python3 scripts/ckpt_to_safetensors_numpy.py \
      --checkpoint models/speechbrain-ecapa-voxceleb/embedding_model.ckpt \
      --output models/speaker-ecapa-voxceleb.safetensors
"""
import argparse
import collections
import json
import pickle
import struct
import sys
import zipfile

import numpy as np

# storage class name -> (numpy dtype, safetensors dtype string)
STORAGE_DTYPE = {
    "FloatStorage": (np.float32, "F32"),
    "HalfStorage": (np.float16, "F16"),
    "DoubleStorage": (np.float64, "F64"),
    "BFloat16Storage": (None, "BF16"),
    "LongStorage": (np.int64, "I64"),
    "IntStorage": (np.int32, "I32"),
    "ByteStorage": (np.uint8, "U8"),
}


class StorageType:
    def __init__(self, name):
        self.name = name


class StorageStub:
    def __init__(self, key, storage_name, numel):
        self.key = key
        self.storage_name = storage_name
        self.numel = numel


class TensorStub:
    def __init__(self, storage, offset, size, stride):
        self.storage = storage
        self.offset = offset
        self.size = tuple(size)
        self.stride = tuple(stride)


def _rebuild_tensor_v2(storage, offset, size, stride, *rest):
    return TensorStub(storage, offset, size, stride)


def _rebuild_parameter(tensor, *rest):
    return tensor


def _noop(*args, **kwargs):
    return None


def make_unpickler(buf):
    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if name.endswith("Storage"):
                return StorageType(name)
            if name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if name == "_rebuild_parameter":
                return _rebuild_parameter
            if name == "OrderedDict":
                return collections.OrderedDict
            try:
                return super().find_class(module, name)
            except Exception:
                return _noop

        def persistent_load(self, pid):
            # pid = ("storage", StorageType, key, location, numel)
            _, storage_type, key, _location, numel = pid
            return StorageStub(str(key), storage_type.name, numel)

    return U(buf)


def load_state_dict(path):
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        pkl_name = next(n for n in names if n.endswith("data.pkl"))
        prefix = pkl_name[: -len("data.pkl")]
        with zf.open(pkl_name) as f:
            obj = make_unpickler(f).load()

        # Unwrap to the tensor dict.
        state = obj
        if isinstance(state, dict):
            for k in ("state_dict", "model", "embedding_model"):
                if isinstance(state.get(k), dict):
                    state = state[k]
                    break

        out = collections.OrderedDict()
        for name, t in state.items():
            if not isinstance(t, TensorStub):
                continue
            np_dtype, _ = STORAGE_DTYPE[t.storage.storage_name]
            if np_dtype is None:
                continue  # skip unsupported (e.g. bf16) - ECAPA is f32
            raw = zf.read(f"{prefix}data/{t.storage.key}")
            flat = np.frombuffer(raw, dtype=np_dtype)
            itemsize = flat.itemsize
            if t.stride and t.size:
                strided = np.lib.stride_tricks.as_strided(
                    flat[t.offset:],
                    shape=t.size,
                    strides=[s * itemsize for s in t.stride],
                )
                arr = np.ascontiguousarray(strided)
            else:
                arr = flat.reshape(t.size) if t.size else flat
            out[name] = np.ascontiguousarray(arr, dtype=np.float32)
        return out


def write_safetensors(tensors, path):
    header = {}
    offset = 0
    blobs = []
    for name, arr in tensors.items():
        b = arr.tobytes(order="C")
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(b)],
        }
        offset += len(b)
        blobs.append(b)
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(hjson)) % 8
    hjson += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for b in blobs:
            f.write(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dump-keys", action="store_true")
    args = ap.parse_args()

    tensors = load_state_dict(args.checkpoint)
    if args.dump_keys:
        for name, arr in tensors.items():
            print(f"{name}\t{tuple(arr.shape)}\t{arr.dtype}")
    write_safetensors(tensors, args.output)
    total = sum(a.size for a in tensors.values())
    print(f"wrote {len(tensors)} tensors ({total} floats) to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
