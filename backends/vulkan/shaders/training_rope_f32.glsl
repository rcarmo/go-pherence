#version 450
// Table-driven adjacent-pair R or R^T; one invocation owns both outputs.
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X {float xv[];};
layout(set=0,binding=1) readonly buffer T {float tv[];};
layout(set=0,binding=2) writeonly buffer O {float result[];};
layout(push_constant) uniform Args {uint rows;uint heads;uint dim;uint mode;} a;
void main(){
 uint pair=gl_GlobalInvocationID.x;if(pair>=a.rows*a.heads*(a.dim/2u))return;
 uint i=2u*pair;uint row=i/(a.heads*a.dim);uint col=i%a.dim;
 float c=tv[row*a.dim+col];float s=tv[row*a.dim+col+1u];if(a.mode==1u)s=-s;
 precise float real=xv[i]*c-xv[i+1u]*s;precise float imag=xv[i+1u]*c+xv[i]*s;
 result[i]=real;result[i+1u]=imag;
}
