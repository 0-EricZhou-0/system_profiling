#include "session_metadata_writer.h"

#include "session_metadata.pb.h"

#include <fstream>
#include <iostream>

namespace cupti_profiler {
namespace internal {

void WriteSessionMetadata(const std::string& path, const SessionMetadata& meta) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "Failed to open session metadata output: " << path << "\n";
        return;
    }
    if (!meta.SerializeToOstream(&out)) {
        std::cerr << "Failed to serialize SessionMetadata to " << path << "\n";
        return;
    }
    std::cout << "[Session] Wrote metadata to " << path << "\n";
}

} // namespace internal
} // namespace cupti_profiler
