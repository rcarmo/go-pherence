#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer Q {float q[];};
layout(set=0,binding=1) readonly buffer K {float k[];};
layout(set=0,binding=2) readonly buffer V {float v[];};
layout(set=0,binding=3) readonly buffer P {float probability[];};
layout(set=0,binding=4) readonly buffer G {float dout[];};
// mode1 consumes dP; mode2/3 consume scaled dScore; other modes ignore aux.
layout(set=0,binding=5) readonly buffer A {float aux[];};
layout(set=0,binding=6) writeonly buffer O {float result[];};
layout(push_constant) uniform Args {uint rows;uint heads;uint dim;uint context;uint mode;float scale;} a;
bool keyAllowed(uint query,uint key){return key<query+1u&&(a.context==0u||query-key<a.context);}
void main(){
 uint i=gl_GlobalInvocationID.x;
 uint width=a.heads*a.dim;
 precise float sum=0.0;
 if(a.mode==0u){
  if(i>=a.heads*a.rows*a.rows)return;
  uint key=i%a.rows;uint query=(i/a.rows)%a.rows;uint head=i/(a.rows*a.rows);
  if(keyAllowed(query,key)) for(uint d=0u;d<a.dim;d++) sum=sum+dout[query*width+head*a.dim+d]*v[key*width+head*a.dim+d];
  result[i]=sum;
 }else if(a.mode==1u){
  // One invocation owns a complete softmax derivative query/head row.
  if(i>=a.heads*a.rows)return;
  uint query=i%a.rows;uint base=i*a.rows;
  for(uint key=0u;key<query+1u;key++) if(keyAllowed(query,key)) sum=sum+probability[base+key]*aux[base+key];
  for(uint key=0u;key<a.rows;key++){
   precise float value=0.0;
   if(keyAllowed(query,key))value=(probability[base+key]*(aux[base+key]-sum))*a.scale;
   result[base+key]=value;
  }
 }else{
  if(i>=a.rows*width)return;
  uint row=i/width;uint head=(i%width)/a.dim;uint d=i%a.dim;
  if(a.mode==2u){
   for(uint key=0u;key<row+1u;key++)if(keyAllowed(row,key))sum=sum+aux[(head*a.rows+row)*a.rows+key]*k[key*width+head*a.dim+d];
  }else{
   // Fixed key: only queries at/after key and within context participate.
   for(uint query=row;query<a.rows;query++)if(keyAllowed(query,row)){
    uint idx=(head*a.rows+query)*a.rows+row;
    if(a.mode==3u)sum=sum+aux[idx]*q[query*width+head*a.dim+d];
    else sum=sum+probability[idx]*dout[query*width+head*a.dim+d];
   }
  }
  result[i]=sum;
 }
}
