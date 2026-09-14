#!/usr/bin/env python3
"""Explicit offline30s pinned pipeline oracle. No service/network; local inputs only.

Constructs source models and uses already-converted safetensors. Raw segmentation
metadata is read weights-only with three pinned safe types. Writes source arrays
and reference annotations for comparison, never for Go runtime inference.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import wave

HEX64 = re.compile(r"^[0-9a-f]{64}$")
URI = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")

SOURCE_HASHES = {
    "core/plda.py": "29674eb850f1a96ebd2c56a4a474801351b84129a02c17c5214258e5f6cdc809",
    "utils/vbx.py": "a8c644feea4b381f9c1e7da72e0e47775c1fd482067e686801ddc16e5cac3c0e",
    "core/inference.py": "c29f525c93a1a5c6bd0a9475ae4fb8c73ce9b48cf9735301302eb99886ad7d41",
    "pipelines/speaker_diarization.py": "cbb358abedef5042fcc71bb970b12a2936a16686be4bda15691a224f172656fd",
    "pipelines/speaker_verification.py": "af1d16a03b482dc5509431d113acbbb1466b75c35d8eaa64d4a8bb593e293c28",
    "pipelines/clustering.py": "6031fb7c21277a7e9901ef2cdaed7d5cd69f7ef45508dc4b45e82ce0da3c8fba",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segmentation-dir", required=True)
    p.add_argument("--embedding-dir", required=True)
    p.add_argument("--raw-segmentation", required=True)
    p.add_argument("--plda-dir", required=True)
    p.add_argument("--public-wav", required=True)
    p.add_argument("--public-wav-sha256", default="c319b4abca767b124e41432d364fd7df006cb26bb79d09326c487d606a134e6e")
    p.add_argument("--samples", type=int, default=480000)
    p.add_argument("--uri", default="sample")
    p.add_argument("--num-speakers", type=int)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def checked(path, expected):
        if sha(path) != expected:
            raise ValueError(f"hash mismatch: {path}")
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("output exists")
    for directory, digest in [(args.segmentation_dir,"bc8af5359abf662c93646fd00fe29bec02ab4e5998cef70a92be45af454ff1e0"),
                              (args.embedding_dir,"b6b6081921bbe1b22db78a67b4692c9a5fc0434b0cfa419b818d056e5a8f8ffb")]:
        checked(Path(directory)/"manifest.json",digest)
        manifest=json.loads((Path(directory)/"manifest.json").read_text())
        for name, info in manifest["files"].items():
            checked(Path(directory)/name,info["sha256"])
    checked(args.raw_segmentation,"7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e")
    if not HEX64.fullmatch(args.public_wav_sha256) or not URI.fullmatch(args.uri) or not 1 <= args.samples <= 160000 + 127 * 16000 or args.num_speakers is not None and not 1 <= args.num_speakers <= 64:
        raise ValueError("invalid PCM identity/geometry")
    checked(args.public_wav,args.public_wav_sha256)
    for name,digest in [("xvec_transform.npz","325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f"),("plda.npz","9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255")]:
        checked(Path(args.plda_dir)/name,digest)
    import torch
    import numpy as np
    import pyannote.audio
    from safetensors.torch import load_file
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    from pyannote.audio.models.segmentation.PyanNet import PyanNet
    from pyannote.audio.models.embedding.wespeaker import WeSpeakerResNet34
    from pyannote.audio.core.plda import PLDA
    from pyannote.audio.pipelines import SpeakerDiarization
    base=Path(pyannote.audio.__file__).parent
    for name,digest in SOURCE_HASHES.items():
        checked(base/name,digest)
    # Verify constituent sources from both conversion manifests as well.
    embedding_manifest=json.loads((Path(args.embedding_dir)/"manifest.json").read_text())
    for name,digest in embedding_manifest["source_hashes"].items():
        checked(base/name,digest)
    segmentation_manifest=json.loads((Path(args.segmentation_dir)/"manifest.json").read_text())
    for name,key in [("models/segmentation/PyanNet.py","pyannet"),("models/blocks/sincnet.py","sincnet")]:
        checked(base/name,segmentation_manifest["source_hashes"][key])
    if torch.version.git_version!="08187d9e0fba026dc8217405802ab5381dc88d90":
        raise ValueError("Torch revision changed")
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.mkldnn.enabled=False;torch.manual_seed(0)
    expected={"pyannote.audio.core.task."+n for n in ("Problem","Resolution","Specifications")}
    if set(torch.serialization.get_unsafe_globals_in_checkpoint(args.raw_segmentation))!=expected:
        raise ValueError("checkpoint globals changed")
    with torch.serialization.safe_globals([Problem,Resolution,Specifications]):
        metadata=torch.load(args.raw_segmentation,map_location="cpu",weights_only=True)
    seg=PyanNet(**metadata["hyper_parameters"])
    seg.specifications=metadata["pyannote.audio"]["specifications"];seg.build()
    seg.load_state_dict(load_file(str(Path(args.segmentation_dir)/"segmentation.safetensors")),strict=True);seg.eval()
    emb=WeSpeakerResNet34().eval()
    emb.load_state_dict(load_file(str(Path(args.embedding_dir)/"embedding.safetensors")),strict=True)
    pipeline=SpeakerDiarization(segmentation=seg,embedding=emb,plda=PLDA(Path(args.plda_dir)/"xvec_transform.npz",Path(args.plda_dir)/"plda.npz",lda_dimension=128),
                                segmentation_step=.1,embedding_exclude_overlap=True,embedding_batch_size=1,segmentation_batch_size=1)
    pipeline.instantiate({"segmentation":{"min_duration_off":0.},"clustering":{"threshold":.6,"Fa":.07,"Fb":.8}})
    output.mkdir(parents=True)
    artifacts={}
    def hook(step,artifact=None,**kwargs):
        if step in ("segmentation","embeddings") and artifact is not None:
            values=np.asarray(artifact.data if hasattr(artifact,"sliding_window") else artifact)
            np.save(output/(step+".npy"),values,allow_pickle=False)
            artifacts[step]=dict(shape=list(values.shape),sha256=sha(output/(step+".npy")))
    with wave.open(args.public_wav,"rb") as w:
        if (w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getnframes(),w.getcomptype())!=(1,2,16000,args.samples,"NONE"):raise ValueError("PCM contract")
        raw=w.readframes(args.samples)
        if len(raw) != args.samples * 2: raise ValueError("short PCM read")
        pcm=torch.from_numpy(np.frombuffer(raw,dtype="<i2").copy()).float()[None]/32768
    with torch.no_grad():
        count = {} if args.num_speakers is None else {"num_speakers": args.num_speakers}
        result=pipeline({"waveform":pcm,"sample_rate":16000,"uri":args.uri},min_speakers=1,max_speakers=64,hook=hook,**count)
    def turns(annotation):
        return [dict(start=float(s.start),end=float(s.end),speaker=str(label)) for s,_,label in annotation.itertracks(yield_label=True)]
    data=dict(schema=1,model_revision="3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",source_hashes=SOURCE_HASHES,
              threads=1,mkldnn=False,batch_sizes=1,constrained=True,minimum_embedding_samples=pipeline._embedding.min_num_samples,
              full=turns(result.speaker_diarization),exclusive=turns(result.exclusive_speaker_diarization),artifacts=artifacts)
    if args.num_speakers is not None: data["num_speakers"] = args.num_speakers
    (output/"reference.json").write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps(data))


if __name__=="__main__":main()
