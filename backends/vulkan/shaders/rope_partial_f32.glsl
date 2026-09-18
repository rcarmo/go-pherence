#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer Q { float q[]; };
layout(set=0, binding=1) readonly buffer CosSin { float cs[]; };
// Matches VkRoPEPartialF32's existing interleaved frequency/push API.
layout(push_constant) uniform P { uint pos; uint nHeads; uint headDim; uint rotHalf; };
void main() {
    uint pair = gl_GlobalInvocationID.x;
    uint head = pair / rotHalf;
    if (head >= nHeads) return;
    uint hf = pair % rotHalf;
    uint frequency = (pos * rotHalf + hf) * 2;
    float c = cs[frequency];
    float s = cs[frequency + 1];
    uint first = head * headDim + hf;
    uint second = first + rotHalf;
    // One invocation owns BOTH components; no invocation reads a value that
    // another writes. The host rejects overlap between Q and CosSin.
    float q0 = q[first];
    float q1 = q[second];
    q[first] = q0 * c - q1 * s;
    q[second] = q0 * s + q1 * c;
}
