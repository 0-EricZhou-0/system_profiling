#include "session_metadata_writer.h"

#include "session_metadata.pb.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace cupti_profiler {
namespace internal {

// Atomic write: serialize to <path>.tmp in the same directory, then rename(2)
// onto the final path. A reader tailing the file (e.g. a live visualizer) can
// observe only the previous complete file or the new complete file — never a
// torn intermediate state.
void WriteSessionMetadata(const std::string& path, const SessionMetadata& meta) {
    const std::string tmpPath = path + ".tmp";

    {
        std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
        if (!out) {
            std::cerr << "Failed to open session metadata output: " << tmpPath << "\n";
            return;
        }
        if (!meta.SerializeToOstream(&out)) {
            std::cerr << "Failed to serialize SessionMetadata to " << tmpPath << "\n";
            return;
        }
    } // ofstream destructor flushes + closes before rename

    std::error_code ec;
    std::filesystem::rename(tmpPath, path, ec);
    if (ec) {
        std::cerr << "Failed to rename " << tmpPath << " -> " << path
                  << ": " << ec.message() << "\n";
        std::filesystem::remove(tmpPath, ec);
        return;
    }
    std::cout << "[Session] Wrote metadata to " << path << "\n";
}

} // namespace internal
} // namespace cupti_profiler
