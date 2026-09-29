#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer X { float x[]; };
layout(set=0, binding=1) readonly buffer CosSin { float cs[]; };
// Time-major X[rows, heads, headDim]; CS[rows, rotHalf, cos/sin].
layout(push_constant) uniform P { uint rows; uint heads; uint headDim; uint rotHalf; };
void main() {
    uint pair = gl_GlobalInvocationID.x;
    uint pairs = rows * heads * rotHalf;
    if (pair >= pairs) return;
    uint dim = pair % rotHalf;
    uint head = (pair / rotHalf) % heads;
    uint row = pair / (rotHalf * heads);
    uint frequency = (row * rotHalf + dim) * 2;
    float c = cs[frequency];
    float s = cs[frequency + 1];
    uint first = (row * heads + head) * headDim + dim;
    uint second = first + rotHalf;
    float a = x[first];
    float b = x[second];
    x[first] = a * c - b * s;
    x[second] = a * s + b * c;
}
