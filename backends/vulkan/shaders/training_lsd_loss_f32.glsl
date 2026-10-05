#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer P {float prediction[];};
layout(set=0,binding=1) readonly buffer D {float desired[];};
layout(set=0,binding=2) readonly buffer L {float logvar[];};
layout(set=0,binding=3) writeonly buffer O {float result[];};
layout(push_constant) uniform Args {uint channels;uint mode;} a;
void main(){
 uint i=gl_GlobalInvocationID.x;
 if(a.mode==2u){
  if(i!=0u)return;float invalid=0.0,cutoff=logvar[0];
  for(uint j=0u;j<a.channels;j++){
   precise float d=prediction[j]-desired[j];
   if(!(abs(prediction[j])<=3.402823466e38))invalid=1.0;
   if(!(abs(desired[j])<=3.402823466e38))invalid=1.0;
   if(!(abs(d)<=cutoff))invalid=1.0;
  }
  result[0]=invalid;return;
 }
 precise float scale=exp(logvar[0])/float(a.channels);
 if(a.mode==1u){if(i>=a.channels)return;precise float difference=prediction[i]-desired[i];result[i]=(2.0*difference)*scale;return;}
 if(i!=0u)return;
 precise float square=0.0,correction=0.0;
 for(uint j=0u;j<a.channels;j++){precise float d=prediction[j]-desired[j];precise float term=d*d-correction;precise float next=square+term;correction=(next-square)-term;square=next;}
 precise float normalized=square*scale;
 result[0]=square;result[1]=normalized-logvar[0];result[2]=normalized-1.0;
}
