// Independent opt-in oracle using the pinned whisper.cpp public VAD API.
// Build/link against the retained whisper.cpp source/library after obtaining a
// CPU window. This helper is never called by native Go inference or unit tests.
// Usage: oracle MODEL RAW_F32 OUTPUT.json (mono16k little-endian F32, <=60s).
#include "whisper.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    try {
        if (argc != 4) throw std::runtime_error("require MODEL RAW_F32 OUTPUT.json");
        const uint32_t endian = 1;
        if (*reinterpret_cast<const unsigned char *>(&endian) != 1) throw std::runtime_error("little-endian host required");
        std::ifstream input(argv[2], std::ios::binary);
        if (!input) throw std::runtime_error("cannot open PCM");
        input.seekg(0, std::ios::end);
        const auto length = input.tellg();
        if (length <= 0 || length % 4 || length > 16000u * 60u * 4u) throw std::runtime_error("invalid PCM size");
        input.seekg(0, std::ios::beg);
        std::vector<char> bytes(static_cast<size_t>(length));
        input.read(bytes.data(), length);
        if (!input) throw std::runtime_error("incomplete PCM read");
        std::vector<float> samples(bytes.size()/4);
        std::memcpy(samples.data(), bytes.data(), bytes.size());
        for (float x : samples) if (!std::isfinite(x)) throw std::runtime_error("nonfinite PCM");
        auto params = whisper_vad_default_context_params();
        params.n_threads = 1;
        params.use_gpu = false;
        std::unique_ptr<whisper_vad_context, decltype(&whisper_vad_free)> owner(
            whisper_vad_init_from_file_with_params(argv[1], params), whisper_vad_free);
        auto * context = owner.get();
        if (!context) throw std::runtime_error("VAD model load failed");
        if (!whisper_vad_detect_speech(context, samples.data(), static_cast<int>(samples.size())))
            throw std::runtime_error("VAD inference failed");
        auto options = whisper_vad_default_params();
        std::unique_ptr<whisper_vad_segments, decltype(&whisper_vad_free_segments)> spans(
            whisper_vad_segments_from_probs(context, options), whisper_vad_free_segments);
        auto * segments = spans.get();
        if (!segments) throw std::runtime_error("segmentation failed");
        std::ofstream output(argv[3], std::ios::binary);
        if (!output) throw std::runtime_error("cannot write output");
        output.precision(9);
        output << "{\"schema\":1,\"source_revision\":\"c44b60b8053bbf2a5c1e014f11323fb3f2485177\",\"model_sha256\":\"2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987\",\"samples\":" << samples.size() << ",\"probabilities\":[";
        const int count = whisper_vad_n_probs(context);
        const float * scores = whisper_vad_probs(context);
        for (int i=0; i<count; ++i) {
            if (i) output << ',';
            if (!std::isfinite(scores[i])) throw std::runtime_error("nonfinite reference probability");
            output << scores[i];
        }
        output << "],\"segments_centiseconds\":[";
        for (int i=0; i<whisper_vad_segments_n_segments(segments); ++i) {
            if (i) output << ',';
            output << '[' << whisper_vad_segments_get_segment_t0(segments,i) << ',' << whisper_vad_segments_get_segment_t1(segments,i) << ']';
        }
        output << "]}\n";
        if (!output) throw std::runtime_error("output write failed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "silero-vad-reference: %s\n", error.what());
        return 1;
    }
}
