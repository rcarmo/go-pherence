"""PYTHONPATH=pinned-upstream python this.py tokenizer-directory"""
import json,sys
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from gliner2.processor import SchemaTransformer
p=SchemaTransformer(tokenizer=PreTrainedTokenizerFast(tokenizer_file=str(Path(sys.argv[1])/'tokenizer.json'),pad_token='[PAD]',unk_token='[UNK]',cls_token='[CLS]',sep_token='[SEP]'),token_pooling='first')
text='Ada Lovelace lived in London.'
cases=[]
for marker,parent in [('[E]','entities'),('[L]','sentiment')]:
 labels=['person','location']
 prompt='Choose [E] carefully'
 descriptions={'location':'A place [L] marker','person':'A human'}
 examples=[('Ada is [E] here','person'),('London','location')]
 tokens=p._transform_schema(parent,labels,marker,prompt=prompt,label_descriptions=descriptions,examples=examples,example_mode='both')
 words=p._tokenize_text(text)
 out=p._format_input_with_mapping([tokens],words)
 cases.append({'parent':parent,'marker':marker,'labels':labels,'prompt':prompt,'descriptions':[{'label':k,'text':v} for k,v in descriptions.items()], 'examples':[{'input':a,'label':b} for a,b in examples], 'text':text,'ids':out['input_ids'],'queries':out['schema_special_positions'][0][1:],'positions':out['text_word_first_positions']})
Path(__file__).with_name('schema_reference.json').write_text(json.dumps(cases,indent=2)+'\n')
