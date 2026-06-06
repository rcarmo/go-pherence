import capstone as cap, time
so="/usr/lib/libspacemit_ep.so.2.0.2+rc5"
OFF,SIZE,ADDR=0xfc260,0x2ff7d4,0xfc260
data=open(so,'rb').read()[OFF:OFF+SIZE]
md=cap.Cs(cap.CS_ARCH_RISCV, cap.CS_MODE_RISCV64|cap.CS_MODE_RISCVC)
TARGETS={0xC0046309:"TCM_ACQUIRE",0x80046307:"TCM_QUERY"}
t0=time.time()
pos=0; decoded=0; regwin={}  # rolling: addr->(reg,val)
# We sweep linearly, skipping undecodable bytes; track lui+addiw immediates in a small window
recent=[]  # last few (addr,mnem,op)
hits=0
while pos < SIZE:
    got=False
    for ins in md.disasm(data[pos:pos+4], ADDR+pos, count=1):
        got=True; decoded+=1
        m,o=ins.mnemonic,ins.op_str; p=o.split(', ')
        try:
            if m=='lui' and len(p)==2: regwin[p[0]]=(int(p[1],0)<<12)&0xffffffff
            elif m in('addiw','addi') and len(p)==3 and p[1] in regwin:
                regwin[p[0]]=(regwin[p[1]]+int(p[2],0))&0xffffffff
            elif p: regwin.pop(p[0],None)
        except:
            if p: regwin.pop(p[0],None)
        for r,v in list(regwin.items()):
            if v in TARGETS:
                hits+=1; print("*** %s @0x%x reg %s"%(TARGETS[v],ins.address,r)); regwin.pop(r,None)
        pos += ins.size
        break
    if not got: pos += 2  # skip undecodable (RVC alignment)
print("decoded %d insns, %d hits, %.1fs"%(decoded,hits,time.time()-t0))
