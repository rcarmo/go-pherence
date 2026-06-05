import sys,json
def b2u():
    bs=list(range(33,127))+list(range(161,173))+list(range(174,256)); cs=bs[:]; n=0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n+=1
    return {chr(c):b for b,c in zip(bs,cs)}
BD=b2u(); tok=json.load(open('tokenizer.json')); v=tok['model']['vocab']; id2={i:t for t,i in v.items()}
sp={at['id'] for at in tok.get('added_tokens',[])}
ids=[int(x) for x in sys.stdin.read().split()]
s=''.join(id2.get(i,'') for i in ids if i not in sp and i<50257)
print(bytes([BD.get(c,ord(c)&255) for c in s]).decode('utf-8','replace'))
