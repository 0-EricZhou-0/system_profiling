// Internal: append one length-delimited protobuf message to a trace file
// in a single write.
#pragma once

#include <google/protobuf/io/coded_stream.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

namespace cupti_profiler {
namespace internal {

/// Serializes `msg`, then writes varint(size) + bytes with one
/// std::ofstream::write of the whole frame, which libstdc++ passes to
/// the kernel as one write (the stream's buffer is empty after each
/// flush). Streaming through a CodedOutputStream instead wrote the frame
/// in 8 KiB pieces. Returns the message size (0 = not written).
template <class Msg>
size_t WriteDelimitedFrame(const Msg& msg, std::ofstream& out, const char* what) {
    std::string frame;
    const size_t size = msg.ByteSizeLong();
    frame.reserve(size + 10);
    uint8_t len[10];
    uint8_t* end = google::protobuf::io::CodedOutputStream::WriteVarint32ToArray(
        static_cast<uint32_t>(size), len);
    frame.append(reinterpret_cast<const char*>(len), static_cast<size_t>(end - len));
    if (!msg.AppendToString(&frame)) {
        std::cerr << "Failed to serialize " << what << "\n";
        return 0;
    }
    out.write(frame.data(), static_cast<std::streamsize>(frame.size()));
    return size;
}

} // namespace internal
} // namespace cupti_profiler
