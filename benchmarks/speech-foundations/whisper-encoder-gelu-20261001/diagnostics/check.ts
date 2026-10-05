// CPU oracle: emulate the shader's bit algorithm and compare with exact RNE via Math.fround-like f16 conversion.
function shader(x: number): number { const f=new Float32Array([x]); const u=new Uint32Array(f.buffer); const b=u[0]>>>0; const sign=(b&0x80000000)>>>0; const e=(b>>>23)&0xff; let r=0;
 if(e===102){ if((b&0x7fffff)!==0) r=0x33800000; }
 if(e>=103){ let shift=13; if(e<113) shift=13+(113-e); const mag=(b&0x7fffffff)>>>0; const lsb=(mag>>>shift)&1; const bias=((1<<(shift-1))>>>0)-1+lsb; r=(((mag+bias)>>>0)&((0xffffffff<<shift)>>>0))>>>0; }
 if(r>=0x47800000) r=0x7f800000; u[0]=(sign|r)>>>0; return f[0]; }
// reference: exact RNE to f16 using Float16Array if available
const ref=(x:number)=>{ const h=new Float16Array([x]); return h[0]; };
let bad=0,n=0; const f=new Float32Array(1), u=new Uint32Array(f.buffer);
for(let i=0;i<4e6;i++){ u[0]=(Math.random()*0x100000000)>>>0; const e=(u[0]>>>23)&0xff; if(e>142||e<90) { if (Math.random()<0.5) continue; u[0]=(u[0]&0x807fffff)|((90+Math.floor(Math.random()*52))<<23); } const x=f[0]; if(!isFinite(x)) continue; n++; const a=shader(x), r=ref(x); if(!(a===r || (a===0&&r===0))) { if(bad<5) console.log(x,a,r); bad++; } }
// edge cases
for (const x of [2**-25, -(2**-25), 2**-24, 2**-25*1.5, 2**-14, 2**-14*(1-2**-12), 65504, 1+2**-11, 1+3*2**-11, 0, -0]) { const a=shader(x), r=ref(x); if(a!==r){console.log('edge',x,a,r);bad++;} n++; }
console.log('checked',n,'bad',bad);
