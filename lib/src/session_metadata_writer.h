// Internal: write a single SessionMetadata message to a .pb file.
// Not length-delimited — the file holds exactly one message.
#pragma once

#include <string>

class SessionMetadata;

namespace cupti_profiler {
namespace internal {

void WriteSessionMetadata(const std::string& path, const SessionMetadata& meta);

} // namespace internal
} // namespace cupti_profiler
