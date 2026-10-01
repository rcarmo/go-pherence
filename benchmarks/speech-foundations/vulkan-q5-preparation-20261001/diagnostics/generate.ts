const s=await Bun.file('projects/go-pherence-transcribe-web/backends/vulkan/shaders/linear_q5_decode4_f32.glsl').text();const a=s.indexOf('  // 4 lanes'),b=s.indexOf('  barrier();',a);if(a<0||b<0)throw Error('decode framing');
for(const lanes of [4,8,16]){
 let decode=`  // Prepared9words: exactF32scale plus32 signed Q5 values.\n  for(uint id=lane;id<${64*lanes}u;id+=256u){uint r=id/${lanes}u,chunk=id%${lanes}u;float d=0.0;uint offset=0u;if(colBase+r<outDim){offset=((colBase+r)*(inDim/32u)+base/32u)*9u;d=uintBitsToFloat(weight[offset]);}\n`;
 for(let j=0;j<32/lanes;j++){decode+=`   {uint index=chunk*${32/lanes}u+${j}u;uint bits=0;if(colBase+r<outDim)bits=weight[offset+1u+index/4u];uint byteValue=(bits>>((index%4u)*8u))&255u;float q=float(byteValue&127u)-float(byteValue&128u);wt[r*32u+index]=d*q;}\n`}
 decode+='  }\n';await Bun.write(`tmp/whisper-q5-predecode-20261001/signed${lanes}.glsl`,s.slice(0,a)+decode+s.slice(b));
}
