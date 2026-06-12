import capstone as cap
so="/usr/lib/libspacemit_ep.so.2.0.2+rc5"
ADDR,OFF,SIZE=0xfc260,0xfc260,0x2ff7d4
data=open(so,'rb').read()[OFF:OFF+SIZE]
md=cap.Cs(cap.CS_ARCH_RISCV, cap.CS_MODE_RISCV64|cap.CS_MODE_RISCVC)
insns=list(md.disasm(data, ADDR))
print("disassembled %d insns"%len(insns))
TARGETS={0xC0046309:"TCM_ACQUIRE",0x80046307:"TCM_QUERY",0x40046308:"TCM_q8",0xc0086301:"alt"}
regval={}
hits=0
for i,ins in enumerate(insns):
    m,o=ins.mnemonic,ins.op_str
    p=o.split(', ')
    try:
        if m=='lui' and len(p)==2: regval[p[0]]=(int(p[1],0)<<12)&0xffffffff
        elif m in('addiw','addi','c.addiw') and len(p)==3 and p[1] in regval:
            regval[p[0]]=(regval[p[1]]+int(p[2],0))&0xffffffff
        elif m in('addiw','addi') and len(p)==2 and p[0] in regval:
            regval[p[0]]=(regval[p[0]]+int(p[1],0))&0xffffffff
        elif len(p)>=1: regval.pop(p[0],None)
    except: regval.pop(p[0] if p else '',None)
    for r,v in list(regval.items()):
        if v in TARGETS:
            hits+=1; print("\n*** %s (0x%x) in %s @ 0x%x ***"%(TARGETS[v],v,r,ins.address))
            for j in range(max(0,i-2),min(len(insns),i+8)):
                print("  %x: %s %s"%(insns[j].address,insns[j].mnemonic,insns[j].op_str))
            del regval[r]
print("\ntotal hits:",hits)
