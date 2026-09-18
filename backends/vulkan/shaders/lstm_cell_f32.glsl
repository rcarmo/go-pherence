#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer Gates { float gates[]; };
layout(set=0,binding=1) readonly buffer CellIn { float cell_in[]; };
layout(set=0,binding=2) buffer HiddenOut { float hidden_out[]; };
layout(set=0,binding=3) buffer CellOut { float cell_out[]; };
layout(push_constant) uniform P { uint hidden; };

// One PyTorch-IFGO LSTM state update over caller-precombined affine gates.
// Branch-stable sigmoid/tanh formulas avoid exponent overflow for finite inputs.
float stable_sigmoid(float value) {
    if (value >= 0.0) return 1.0 / (1.0 + exp(-value));
    float e = exp(value);
    return e / (1.0 + e);
}
float stable_tanh(float value) {
    if (value >= 0.0) {
        float e = exp(-2.0 * value);
        return (1.0 - e) / (1.0 + e);
    }
    float e = exp(2.0 * value);
    return (e - 1.0) / (e + 1.0);
}
void main() {
    uint i=gl_GlobalInvocationID.x;
    if (i<hidden) {
        float inputGate=stable_sigmoid(gates[i]);
        float forgetGate=stable_sigmoid(gates[hidden+i]);
        float candidate=stable_tanh(gates[2*hidden+i]);
        float outputGate=stable_sigmoid(gates[3*hidden+i]);
        precise float nextCell=forgetGate*cell_in[i] + inputGate*candidate;
        cell_out[i]=nextCell;
        hidden_out[i]=outputGate*stable_tanh(nextCell);
    }
}
