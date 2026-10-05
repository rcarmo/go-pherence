#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X {float xv[];};
layout(set=0,binding=1) readonly buffer B {float bv[];};
layout(set=0,binding=2) writeonly buffer O {float result[];};
layout(push_constant) uniform Args {uint rows;uint width;uint parts;uint slot;uint mode;} a;
void main(){
 uint i=gl_GlobalInvocationID.x;
 if(a.mode==0u){if(i>=a.rows*a.width)return;result[i]=xv[(i/a.width*a.parts+a.slot)*a.width+i%a.width];}
 else if(a.mode==1u){if(i>=a.rows*a.parts*a.width)return;uint col=i%a.width;uint part=(i/a.width)%a.parts;uint row=i/(a.parts*a.width);result[i]=part==a.slot?xv[row*a.width+col]:0.0;}
 else if(a.mode==2u){if(i>=a.rows*a.width)return;result[i]=xv[i]*bv[i%a.width];}
 else{if(i>=a.width)return;precise float sum=0.0;for(uint row=0u;row<a.rows;row++)sum+=xv[row*a.width+i]*bv[row*a.width+i];result[i]=sum;}
}
