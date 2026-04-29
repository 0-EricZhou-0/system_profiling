#include "disk_readers.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace cupti_profiler {
namespace internal {

std::vector<DiskStatSnapshot> ReadDiskStats(const std::vector<std::string>& devices) {
    std::vector<DiskStatSnapshot> results;
    std::ifstream f("/proc/diskstats");
    if (!f) return results;

    std::string line;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        uint64_t major, minor;
        std::string devName;
        iss >> major >> minor >> devName;

        // Check if this device is in our list
        bool wanted = false;
        for (const auto& d : devices) {
            if (d == devName) { wanted = true; break; }
        }
        if (!wanted) continue;

        // Fields after device name (1-indexed from the kernel docs):
        //  1: reads completed
        //  2: reads merged
        //  3: sectors read      ← we want this (field index 3, 0-based field[2])
        //  4: read time ms
        //  5: writes completed
        //  6: writes merged
        //  7: sectors written   ← we want this (field index 7, 0-based field[6])
        //  ...
        std::vector<uint64_t> fields;
        uint64_t val;
        while (iss >> val) {
            fields.push_back(val);
        }

        DiskStatSnapshot snap;
        snap.device = devName;
        if (fields.size() > 2) snap.sectorsRead = fields[2];
        if (fields.size() > 6) snap.sectorsWritten = fields[6];
        results.push_back(snap);
    }
    return results;
}

DiskInflightSnapshot ReadDiskInflight(const std::string& device) {
    DiskInflightSnapshot s;
    std::string path = "/sys/block/" + device + "/inflight";
    std::ifstream f(path);
    if (!f) return s;
    f >> s.readInflight >> s.writeInflight;
    return s;
}

PIDIOSnapshot ReadPIDIO(uint32_t pid) {
    PIDIOSnapshot s;
    std::string path = "/proc/" + std::to_string(pid) + "/io";
    std::ifstream f(path);
    if (!f) {
        s.accessible = false;
        return s;
    }

    // Format: key: value (one per line)
    // We want: read_bytes and write_bytes (physical I/O)
    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, 12, "read_bytes: ") == 0) {
            std::istringstream(line.substr(12)) >> s.readBytes;
        } else if (line.compare(0, 13, "write_bytes: ") == 0) {
            std::istringstream(line.substr(13)) >> s.writeBytes;
        }
    }
    return s;
}

} // namespace internal
} // namespace cupti_profiler
