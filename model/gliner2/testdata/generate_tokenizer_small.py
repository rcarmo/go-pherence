"""Generate a self-contained tokenizers oracle with a deliberately ambiguous vocabulary."""
import json
from pathlib import Path
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, AddedToken
vocab=[('[UNK]',0.),('[CLS]',0.),('[SEP]',0.),('[E]',0.),('▁',-1.),('a',-1.),('b',-1.),('ab',-9.),('▁a',-1.5),('é',-1.),('c',-1.),('▁c',-1.5)]
t=Tokenizer(models.Unigram(vocab,unk_id=0,byte_fallback=False))
t.normalizer=normalizers.Sequence([normalizers.Replace(__import__('tokenizers').Regex(r'\s{2,}|[\n\r\t]'),' '),normalizers.NFC(),normalizers.Strip(left=False,right=True)])
t.pre_tokenizer=pre_tokenizers.Sequence([pre_tokenizers.Metaspace(replacement='▁',prepend_scheme='always',split=True)])
t.add_special_tokens([AddedToken(x,special=True,normalized=False) for x in ['[UNK]','[CLS]','[SEP]','[E]']])
t.post_processor=processors.TemplateProcessing(single='[CLS] $A [SEP]',special_tokens=[('[CLS]',1),('[SEP]',2)])
p=Path(__file__).parent
t.save(str(p/'tokenizer_small.json'))
texts=['ab','ab ab','a\u00a0b','a\u2003b','a\u2003\u2003b','cafe\u0301','[E]ab[E]c','', ' \t\n ', '🦊🫠','a\n b']
(p/'tokenizer_small_reference.json').write_text(json.dumps([{'text':x,'ids':t.encode(x).ids} for x in texts],ensure_ascii=False,indent=2)+'\n')
