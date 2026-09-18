"""PYTHONPATH=pinned-upstream python this.py tokenizer-directory"""
import json,sys
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from gliner2.processor import SchemaTransformer
p=SchemaTransformer(tokenizer=PreTrainedTokenizerFast(tokenizer_file=str(Path(sys.argv[1])/'tokenizer.json'),pad_token='[PAD]',unk_token='[UNK]',cls_token='[CLS]',sep_token='[SEP]'))
schemas=[{'parent':'entities','marker':'[E]','labels':['person','location']},{'parent':'sentiment','marker':'[L]','labels':['positive','negative']},{'parent':'lives_in','marker':'[R]','labels':['head','tail']}]
text='Ada Lovelace lived in London.'
tokens=[p._transform_schema(s['parent'],s['labels'],s['marker']) for s in schemas]
x=p._format_input_with_mapping(tokens,p._tokenize_text(text))
data={'text':text,'schemas':schemas,'ids':x['input_ids'],'groups':[g[1:] for g in x['schema_special_positions']],'positions':x['text_word_first_positions']}
Path(__file__).with_name('mixed_schema_reference.json').write_text(json.dumps(data,indent=2)+'\n')
