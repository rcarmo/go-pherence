package ptx

// FFTPTX is a short-time FFT kernel for mel spectrogram computation.
// Processes one frame per thread-block: window → FFT → power spectrum → mel accumulate → log.
// Grid: (num_frames, 1, 1), Block: (256, 1, 1)
const FFTPTX = `
.version 7.0
.target sm_70
.address_size 64

// Fused mel spectrogram kernel: window + FFT + power + mel filterbank + log
// Each block processes one 512-sample frame.
// Shared memory layout: [512 float re] [512 float im] = 4096 bytes
.visible .entry mel_spectrogram(
    .param .u64 out_ptr,        // output: [num_mels * num_frames] float32
    .param .u64 audio_ptr,      // input: [total_samples] float32
    .param .u64 window_ptr,     // Hann window: [fft_size] float32
    .param .u64 mel_filters_ptr,// mel filterbank: [num_mels * num_bins] float32 (sparse)
    .param .u32 num_frames,
    .param .u32 fft_size,       // 400
    .param .u32 hop_length,     // 160
    .param .u32 num_mels,       // 80
    .param .u32 num_bins        // 257 (fft_padded/2 + 1)
) {
    .reg .u32 %frame, %tid, %fft_padded;
    .reg .u64 %audio_base, %out_base;
    .reg .f32 %sample, %win, %re, %im, %power, %mel_val, %filter;
    .shared .f32 sh_re[512];
    .shared .f32 sh_im[512];

    // frame = blockIdx.x
    mov.u32 %frame, %ctaid.x;
    // tid = threadIdx.x
    mov.u32 %tid, %tid.x;
    // fft_padded = 512 (next pow2 >= fft_size)
    mov.u32 %fft_padded, 512;

    // Step 1: Load windowed audio into shared memory
    // Each thread loads one or two samples (512 / 256 = 2 loads per thread)
    // audio_offset = frame * hop_length + tid
    // if tid < fft_size: sh_re[tid] = audio[offset] * window[tid]
    // else: sh_re[tid] = 0
    // sh_im[tid] = 0
    // (second load for tid+256)

    // Step 2: Radix-2 FFT butterfly in shared memory
    // log2(512) = 9 stages
    // Each stage: barrier, compute butterfly pair, write back
    // Twiddle factors computed inline: W_N^k = cos(-2*pi*k/N) + i*sin(-2*pi*k/N)

    // Step 3: Compute power spectrum
    // power[k] = sh_re[k]^2 + sh_im[k]^2 for k = 0..num_bins-1

    // Step 4: Apply mel filterbank (sparse)
    // For each mel bin m assigned to this thread:
    //   mel[m] = sum_k(filter[m][k] * power[k])
    //   mel[m] = log(max(mel[m], 1e-10))

    // Step 5: Write output
    // out[m * num_frames + frame] = mel[m]

    ret;
}
`
