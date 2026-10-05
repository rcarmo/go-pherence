#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X {float x[];};
layout(set=0,binding=1) buffer O {float y[];};
layout(push_constant) uniform P {uint count;};
// Diagnostic encoder-only original GGML Vulkan op_gelu expression.
// Match expression order; do not substitute erf, clamp or approximate further.
float original_gelu(float x){
 const float a=0.044715;
 const float c=0.79788456080286535587989211986876;
 const float val=c*x*(1.0+a*x*x);
 return 0.5*x*(2.0-2.0/(exp(2.0*val)+1.0));
}
void main(){uint i=gl_GlobalInvocationID.x;if(i<count)y[i]=original_gelu(x[i]);}
