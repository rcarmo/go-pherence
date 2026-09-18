"""Regenerate with a Python environment providing torch and transformers.
Reference only; production inference has no Python dependency.
"""
import json
from pathlib import Path
import torch
import transformers
from transformers import DebertaV2Config
from transformers.models.deberta_v2.modeling_deberta_v2 import DisentangledSelfAttention

torch.manual_seed(17)
cfg = DebertaV2Config(hidden_size=8, num_attention_heads=2,
    relative_attention=True, pos_att_type=['c2p', 'p2c'], share_att_key=True,
    position_buckets=4, max_relative_positions=16, max_position_embeddings=16,
    hidden_dropout_prob=0, attention_probs_dropout_prob=0)
m = DisentangledSelfAttention(cfg).eval()
hidden = torch.randn(1, 7, 8)
relative = torch.randn(8, 8)
mask = torch.tensor([[1,1,1,1,1,0,0]], dtype=torch.bool)
attention_mask = mask[:,None,:,None] & mask[:,None,None,:]
with torch.no_grad():
    out, _ = m(hidden, attention_mask, rel_embeddings=relative)
    data = dict(transformers_version=transformers.__version__, heads=2, buckets=4,
        max_position=16, mask=mask[0].tolist(),
        query=m.query_proj(hidden)[0].tolist(), key=m.key_proj(hidden)[0].tolist(),
        value=m.value_proj(hidden)[0].tolist(),
        relative_query=m.query_proj(relative).tolist(),
        relative_key=m.key_proj(relative).tolist(), expected=out[0].tolist())
Path(__file__).with_name('deberta_attention_reference.json').write_text(json.dumps(data, indent=2)+'\n')
