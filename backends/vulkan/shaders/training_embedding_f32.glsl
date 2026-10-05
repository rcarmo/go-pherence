#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X {float xv[];};
layout(set=0,binding=1) readonly buffer IDs {float ids[];};
layout(set=0,binding=2) writeonly buffer O {float result[];};
layout(push_constant) uniform Args {uint rows;uint width;uint vocabulary;uint mode;} a;
void main(){
 uint i=gl_GlobalInvocationID.x;
 if(a.mode==0u){
  if(i>=a.rows*a.width)return;
  float id=ids[i/a.width];
  if(!(id>=0.0&&id<float(a.vocabulary))){result[i]=0.0;return;}
  uint token=uint(id);result[i]=id==float(token)?xv[token*a.width+i%a.width]:0.0;
 }else if(a.mode==3u){
  if(i>=a.rows*a.width)return;
  uint slot=i/a.width,col=i%a.width;
  float token=ids[slot],lo=ids[a.rows+slot],hi=ids[a.rows+slot+1];precise float sum=0.0;
  if(!(token>=0.0)){result[i]=0.0;return;}
  if(!(token<float(a.vocabulary))){result[i]=0.0;return;}
  if(!(lo>=0.0)){result[i]=0.0;return;}
  if(!(hi>=lo)){result[i]=0.0;return;}
  if(!(hi<=float(a.rows))){result[i]=0.0;return;}
  if(token==float(uint(token))){if(lo==float(uint(lo))){if(hi==float(uint(hi))){
   for(uint at=uint(lo);at<uint(hi);at++){
    float row=ids[2u*a.rows+1u+at];
    if(!(row>=0.0))continue;
    if(!(row<float(a.rows)))continue;
    if(row==float(uint(row)))sum+=xv[uint(row)*a.width+col];
   }
   result[i]=sum;return;
  }}}
  result[i]=0.0;
 }else if(a.mode==2u){
  if(i>=a.vocabulary*a.width)return;
  uint token=i/a.width,col=i%a.width;
  float lo=ids[token],hi=ids[token+1];precise float sum=0.0;
  // Separate guards keep conversions after bounds and stay inside the
  // conservative SPIR-V opcode contract (no unordered-not-equal opcode).
  if(!(lo>=0.0)){result[i]=0.0;return;}
  if(!(hi>=lo)){result[i]=0.0;return;}
  if(!(hi<=float(a.rows))){result[i]=0.0;return;}
  if(lo==float(uint(lo))){
   if(hi==float(uint(hi))){
    for(uint at=uint(lo);at<uint(hi);at++){
     float row=ids[a.vocabulary+1u+at];
     if(!(row>=0.0))continue;
     if(!(row<float(a.rows)))continue;
     if(row==float(uint(row)))sum+=xv[uint(row)*a.width+col];
    }
    result[i]=sum;return;
   }
  }
  result[i]=0.0;
 }else{
  if(i>=a.vocabulary*a.width)return;
  uint token=i/a.width,col=i%a.width;precise float sum=0.0;
  for(uint row=0u;row<a.rows;row++){if(ids[row]==float(token))sum+=xv[row*a.width+col];}
  result[i]=sum;
 }
}
